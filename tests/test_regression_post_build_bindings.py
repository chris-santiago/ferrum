"""A regression figure reads or refuses an x/y binding written after its own call.

``regplot``, ``lmplot`` and ``residplot`` build their scatter and fit (or
residual) layers once, at the call, from the ``x=``/``y=`` they were given.
Every door that writes a LATER x/y binding on the returned chart --
``.encode()``, the figure's own ``encode=`` kwarg (which lands on the chart
the same way, after the layers are already built), and ``+`` -- is read, not
dropped: a ``bin=``/``aggregate=`` the fit cannot fold in is refused, a
declared type that differs from the call's own is refused naming the call
door, and a field-less binding is refused naming the column to restate.
A ``scale=``, ``format=``, ``format_type=`` or ``axis=`` on the called
field pushes down onto the figure's own built layers, the later write
always replacing whatever a layer already carries
for that option, so the result is byte-identical to writing the same option
at the call. residplot's y is the one exception: it keeps refusing a
post-build ``scale=``/``format=``, since the response column the call named
is not the residual the figure draws. A binding on any other field (another
variable, or a figure's own drawn column) is unaffected, whatever else the
same call writes -- except flat residplot's x/y, which draw straight from
the chart-level encoding with no layer of their own to shadow it: naming
another column of the data is refused, naming the call as the remedy,
since the one mark would draw that column raw instead of refitting
against it; a column the data lacks still fails the same raw way.
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl
import pytest

import ferrum as fm
from ferrum._entry import EntryRefusal
from ferrum.encoding import X, Y


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "w": [2.0, 3.0, 1.0, 5.0, 4.0, 2.0, 6.0, 3.0],
            "u": [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0],
            "b": [0, 1, 0, 1, 1, 0, 1, 0],
            "g": ["a", "b", "a", "b", "a", "b", "a", "b"],
        }
    )


def _regplot_eager() -> "fm.Chart":
    return fm.regplot(_frame(), x="v", y="w")


def _lmplot_hue() -> "fm.Chart":
    return fm.lmplot(_frame(), x="v", y="w", hue="b")


def _regplot_scatter_false_pending() -> "fm.Chart":
    return fm.regplot(_frame(), x="v", y="w", scatter=False)


def _regplot_scatter_false_dedicated() -> "fm.Chart":
    return fm.regplot(_frame(), x="v", y="w", scatter=False, method="robust")


def _regplot_scatter_false_logistic() -> "fm.Chart":
    return fm.regplot(_frame(), x="v", y="b", scatter=False, method="logistic")


def _regplot_scatter_false_glm() -> "fm.Chart":
    return fm.regplot(_frame(), x="v", y="b", scatter=False, method="glm")


def _lmplot_plain() -> "fm.Chart":
    return fm.lmplot(_frame(), x="v", y="w")


def _residplot_flat() -> "fm.Chart":
    return fm.residplot(_frame(), x="v", y="w", zero_line=False, show_metrics=False)


def _residplot_layered() -> "fm.Chart":
    return fm.residplot(_frame(), x="v", y="w", lowess=True)


def _lmplot_col() -> "fm.Chart":
    return fm.lmplot(_frame(), x="v", y="w", col="g")


_EAGER_ENTRY_FORMS = {
    "regplot": _regplot_eager,
    "lmplot_plain": _lmplot_plain,
    "lmplot_hue": _lmplot_hue,
    "regplot_scatter_false_dedicated": _regplot_scatter_false_dedicated,
    "regplot_scatter_false_logistic": _regplot_scatter_false_logistic,
    "regplot_scatter_false_glm": _regplot_scatter_false_glm,
    "residplot_flat": _residplot_flat,
    "residplot_layered": _residplot_layered,
}

_ALL_FORMS = {**_EAGER_ENTRY_FORMS, "regplot_scatter_false_pending": _regplot_scatter_false_pending}

#: _EAGER_ENTRY_FORMS plus the two forms the channel-options sweep further
#: down covers but the declared-type-mismatch test above does not: a
#: genuinely pending smooth-routed scatter=False regplot (resolved live,
#: at render, through the same push-down this sweep proves) and an
#: lmplot(col=) facet grid. Both measure identically to every eager form
#: for every _CHANNEL_OPTIONS entry, so they widen that sweep without a
#: form-specific carve-out; the pending form's own declared-type quirk
#: (dropped, not refused, with no scatter drawn) is pinned separately by
#: the pending-form tests below instead.
_FULL_SWEEP_FORMS = {
    **_EAGER_ENTRY_FORMS,
    "regplot_scatter_false_pending": _regplot_scatter_false_pending,
    "lmplot_col": _lmplot_col,
}

#: Each form's entry function and its own extra call-time kwargs (beyond
#: x=/y=), shared by the keyword- and encode=-door builders below so the
#: ten forms' call shapes live in one place rather than two parallel
#: if/elif chains.
_FORM_CALL: dict[str, tuple[Callable[..., "fm.Chart"], dict]] = {
    "regplot": (fm.regplot, {}),
    "lmplot_plain": (fm.lmplot, {}),
    "lmplot_hue": (fm.lmplot, {"hue": "b"}),
    "lmplot_col": (fm.lmplot, {"col": "g"}),
    "regplot_scatter_false_pending": (fm.regplot, {"scatter": False}),
    "regplot_scatter_false_dedicated": (fm.regplot, {"scatter": False, "method": "robust"}),
    "regplot_scatter_false_logistic": (fm.regplot, {"scatter": False, "method": "logistic"}),
    "regplot_scatter_false_glm": (fm.regplot, {"scatter": False, "method": "glm"}),
    "residplot_flat": (fm.residplot, {"zero_line": False, "show_metrics": False}),
    "residplot_layered": (fm.residplot, {"lowess": True}),
}


def _build_with_keyword_binding(form_name: str, channel: str, binding: "ChannelBase") -> "fm.Chart":
    """*form_name*'s chart, built with *binding* passed directly as the call's own x=/y= -- the keyword door."""
    entry_fn, extra = _FORM_CALL[form_name]
    y_field = _y_field_of(form_name)
    x_value = binding if channel == "x" else "v"
    y_value = binding if channel == "y" else y_field
    return entry_fn(_frame(), x=x_value, y=y_value, **extra)


def _build_with_encode_kwarg(form_name: str, channel: str, binding: "ChannelBase") -> "fm.Chart":
    """*form_name*'s chart, built with *binding* passed through its own ``encode=`` kwarg."""
    entry_fn, extra = _FORM_CALL[form_name]
    y_field = _y_field_of(form_name)
    return entry_fn(_frame(), x="v", y=y_field, encode={channel: binding}, **extra)


def _entry_of(form_name: str) -> str:
    if form_name.startswith("residplot"):
        return "residplot"
    return "lmplot" if form_name.startswith("lmplot") else "regplot"


#: Every form's own y field: "w" for every form but the two logistic/glm
#: ones, which fit a binary response ("b") instead.
_Y_FIELD_OF: dict[str, str] = {
    "regplot_scatter_false_logistic": "b",
    "regplot_scatter_false_glm": "b",
}


def _y_field_of(form_name: str) -> str:
    return _Y_FIELD_OF.get(form_name, "w")


#: Every scatter=False form with no dedicated scatter layer drawn: the call
#: door drops a declared type the same way the post-build door does, so
#: these share one type= remedy ending (_lmplot_type_remedy_without_scatter).
_NO_SCATTER_DRAWN_FORMS = frozenset(
    {
        "regplot_scatter_false_dedicated",
        "regplot_scatter_false_logistic",
        "regplot_scatter_false_glm",
    }
)


# ---------------------------------------------------------------------------
# A post-build bin=/aggregate=, on every form and either called field, is
# refused with the call-time text.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form_name", sorted(_ALL_FORMS))
@pytest.mark.parametrize("field", ["v", "u"])
@pytest.mark.parametrize("kwarg", ["bin", "aggregate"])
def test_a_post_build_bin_or_aggregate_is_refused_with_the_call_time_text(form_name, field, kwarg):
    build = _ALL_FORMS[form_name]
    chart = build()
    entry = _entry_of(form_name)
    value = True if kwarg == "bin" else "mean"
    post_hoc = chart.encode(x=X(field, **{kwarg: value}))
    with pytest.raises(ValueError) as post_err:
        post_hoc.to_svg()

    # The call-time text, raised directly on a fresh call with the same
    # binding, is the one the post-build backstop must reuse verbatim.
    with pytest.raises(ValueError) as call_err:
        if form_name == "residplot_flat":
            fm.residplot(
                _frame(), x=X(field, **{kwarg: value}), y="w", zero_line=False, show_metrics=False
            )
        elif form_name == "residplot_layered":
            fm.residplot(_frame(), x=X(field, **{kwarg: value}), y="w", lowess=True)
        elif form_name == "lmplot_plain":
            fm.lmplot(_frame(), x=X(field, **{kwarg: value}), y="w")
        elif form_name == "lmplot_hue":
            fm.lmplot(_frame(), x=X(field, **{kwarg: value}), y="w", hue="b")
        elif form_name == "regplot_scatter_false_pending":
            fm.regplot(_frame(), x=X(field, **{kwarg: value}), y="w", scatter=False)
        elif form_name == "regplot_scatter_false_dedicated":
            fm.regplot(
                _frame(), x=X(field, **{kwarg: value}), y="w", scatter=False, method="robust"
            )
        elif form_name == "regplot_scatter_false_logistic":
            fm.regplot(
                _frame(), x=X(field, **{kwarg: value}), y="b", scatter=False, method="logistic"
            )
        elif form_name == "regplot_scatter_false_glm":
            fm.regplot(_frame(), x=X(field, **{kwarg: value}), y="b", scatter=False, method="glm")
        else:
            fm.regplot(_frame(), x=X(field, **{kwarg: value}), y="w")

    assert str(post_err.value) == str(call_err.value)
    assert str(post_err.value).startswith(f"{entry}(")


def test_a_post_build_aggregate_on_y_is_refused_too():
    chart = fm.regplot(_frame(), x="v", y="w")
    with pytest.raises(ValueError) as err:
        chart.encode(y=Y("w", aggregate="mean")).to_svg()
    assert str(err.value) == (
        "regplot(y=Y('w', aggregate='mean')) is not supported: the fit reads the raw response, "
        "so aggregate= would be ignored. Compute the aggregate with "
        ".transform(fm.transform_aggregate(...)) and bind its output column, or drop aggregate=."
    )


# ---------------------------------------------------------------------------
# A declared type that differs from the call's own is refused, naming the
# call as the remedy; the same holds through encode=. scale=/format= push
# down onto the built layers instead; see the push-down tests below.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form_name", sorted(_EAGER_ENTRY_FORMS))
def test_a_declared_type_differing_from_the_call_is_refused(form_name):
    build = _EAGER_ENTRY_FORMS[form_name]
    chart = build()
    entry = _entry_of(form_name)
    field = "v"
    with pytest.raises(ValueError) as err:
        chart.encode(x=X(field, type="T")).to_svg()
    if form_name in _NO_SCATTER_DRAWN_FORMS:
        # With no scatter drawn, the call door drops a declared type too
        # (regplot(x=X('v', type='T'), scatter=False, method="robust")
        # renders byte-identical to the plain form), so naming the call as
        # a remedy would send the caller to a door that also fails.
        ending = "Drop type=: with no scatter drawn, a declared type is never applied."
    else:
        # On x, the call door honors a declared T (the common case, since
        # the call itself declared none), so the remedy names it -- unlike
        # residplot's y under a log scale, covered by its own test below.
        ending = f"Drop type=, or pass it in the call: {entry}(x=fm.X({field!r}, type=...))."
    assert str(err.value) == (
        f"{entry}(x=X({field!r}, type='T')) after the call is not supported: {entry} draws x "
        f"from the x= it was called with, so type= would be ignored. {ending}"
    )


def test_a_declared_type_matching_the_call_is_honored_not_refused():
    """A post-build type that matches the call's own (here, both unset --
    the implicit quantitative default) is not a mismatch, so it reaches
    the output instead of being refused."""
    chart = fm.regplot(_frame(), x="v", y="w")
    plain = chart.to_svg()
    styled = chart.encode(x=X("v", type="Q", title="P")).to_svg()
    assert styled != plain


@pytest.mark.parametrize("form_name", sorted(_EAGER_ENTRY_FORMS))
@pytest.mark.parametrize("door", ["later_dot_encode", "call_time_encode_kwarg"])
def test_a_post_build_title_on_a_declared_temporal_call_is_honored_not_refused(form_name, door):
    """A post-build binding that declares no type of its own (title= alone)
    is a match, not a mismatch, on a call that declared a temporal x --
    byte-identical to the keyword door, on every form, through both
    post-build doors. `Chart._resolve_pending` runs this record more than
    once reaching one render; a later pass must recognize the channel an
    earlier pass already settled, not read its own output (forced to
    type='Q' by `_rebind_positional`, regardless of what the call
    declared) as a fresh caller write that conflicts with the call's own
    declared T."""
    entry_fn, extra = _FORM_CALL[form_name]
    y_field = _y_field_of(form_name)
    keyword = entry_fn(_frame(), x=X("v", type="T", title="Date"), y=y_field, **extra).to_svg()
    if door == "later_dot_encode":
        styled = (
            entry_fn(_frame(), x="v:T", y=y_field, **extra).encode(x=X("v", title="Date")).to_svg()
        )
    else:
        styled = entry_fn(
            _frame(), x="v:T", y=y_field, encode={"x": X("v", title="Date")}, **extra
        ).to_svg()
    assert styled == keyword


@pytest.mark.parametrize("form_name", sorted(_EAGER_ENTRY_FORMS))
@pytest.mark.parametrize("door", ["later_dot_encode", "call_time_encode_kwarg"])
def test_a_post_build_type_matching_a_declared_temporal_call_is_honored(form_name, door):
    """The explicit-match sibling of the test above: a post-build type='T'
    that matches the call's own declared T is honored too, not just an
    unset one -- on every eager form, including the three no-scatter-drawn
    ones. Those three carry a chart-level x/y of their own (the same fix
    that closed the keyword-door axis=None gap), so call_type reads the
    call's own declared T there too, and a matching post-build 'T' is
    honored, byte-identical to the keyword door. An UNDECLARED call
    (implicit Q) refusing a post-build 'T' on these same three forms is
    the different, still-refused case the declared-type-mismatch test
    above covers instead."""
    entry_fn, extra = _FORM_CALL[form_name]
    y_field = _y_field_of(form_name)
    keyword = entry_fn(_frame(), x=X("v", type="T", title="Date"), y=y_field, **extra).to_svg()
    binding = X("v", type="T", title="Date")
    if door == "later_dot_encode":
        styled = entry_fn(_frame(), x="v:T", y=y_field, **extra).encode(x=binding).to_svg()
    else:
        styled = entry_fn(_frame(), x="v:T", y=y_field, encode={"x": binding}, **extra).to_svg()
    assert styled == keyword


# ---------------------------------------------------------------------------
# scale=/format= push down onto the figure's already-built layers, the
# later write always replacing whatever a layer already carries for that
# option -- byte-identical to writing the same option at the call. Every
# door lands a binding the same way: a later .encode(), the figure's own
# encode= kwarg (which lands post-build too), and a + composition's left
# operand.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form_name", sorted(_EAGER_ENTRY_FORMS))
@pytest.mark.parametrize("channel", ["x", "y"])
@pytest.mark.parametrize(
    "kwargs",
    [{"scale": fm.LogScale()}, {"scale": fm.LinearScale(domain=(-5.0, 5.0))}, {"format": ".3f"}],
)
def test_a_post_build_scale_or_format_pushes_down_byte_identical_to_the_keyword_door(
    form_name, channel, kwargs
):
    field = "v" if channel == "x" else _y_field_of(form_name)
    cls = X if channel == "x" else Y
    plain = _EAGER_ENTRY_FORMS[form_name]()
    baseline = plain.to_svg()
    binding = cls(field, **kwargs)

    if _entry_of(form_name) == "residplot" and channel == "y":
        # residplot's y is the one exception: it keeps refusing a
        # post-build scale=/format=, pinned directly by the dedicated
        # residplot tests above.
        with pytest.raises(ValueError):
            plain.encode(**{channel: binding}).to_svg()
        return

    keyword = _build_with_keyword_binding(form_name, channel, binding).to_svg()
    post_build = plain.encode(**{channel: binding}).to_svg()
    assert post_build != baseline
    assert post_build == keyword


@pytest.mark.parametrize("form_name", sorted(_NO_SCATTER_DRAWN_FORMS))
def test_keyword_format_is_honored_on_every_scatter_false_dedicated_method(form_name):
    """5c-2: scatter=False with an explicit method (robust/logistic/glm,
    no scatter layer drawn) still honors a keyword format= on x, the same
    as every other eager form."""
    plain = _build_with_keyword_binding(form_name, "x", "v").to_svg()
    formatted = _build_with_keyword_binding(form_name, "x", X("v", format=".3f")).to_svg()
    assert formatted != plain


@pytest.mark.parametrize("form_name", sorted(_NO_SCATTER_DRAWN_FORMS))
@pytest.mark.parametrize("append_door", ["chart_layer_method", "layers_kwarg"])
def test_a_dedicated_scatter_false_fit_draws_an_appended_layer(form_name, append_door):
    """The three dedicated scatter=False fits now carry a chart-level x/y
    of their own (the same fix that closed the keyword-door axis=None
    gap), so an appended point layer inherits it and draws, byte-identical
    to a `+` composition with the same point chart, the same way the
    pending form, the scatter=True forms and mark_smooth's own layered
    charts already do -- rather than refusing for want of a positional
    channel, as it did before that fix."""
    entry_fn, extra = _FORM_CALL[form_name]
    y_field = _y_field_of(form_name)
    base = entry_fn(_frame(), x="v", y=y_field, **extra)
    plus = (base + fm.Chart(_frame()).mark_point().encode(x="v", y=y_field)).to_svg()
    if append_door == "chart_layer_method":
        appended = base.layer(fm.Layer(mark="point")).to_svg()
    else:
        appended = entry_fn(
            _frame(), x="v", y=y_field, layers=[fm.Layer(mark="point")], **extra
        ).to_svg()
    assert appended == plus


def test_a_post_build_log_scale_on_regplot_x_pushes_down():
    chart = fm.regplot(_frame(), x="v", y="w")
    plain = chart.to_svg()
    keyword = fm.regplot(_frame(), x=X("v", scale=fm.LogScale()), y="w").to_svg()
    post_build = chart.encode(x=X("v", scale=fm.LogScale())).to_svg()
    assert post_build != plain
    assert post_build == keyword


def test_a_post_build_format_type_wins_over_the_calls_own_whichever_spelling_either_side_used():
    """The later write wins across the format_type/formatType alias too,
    not just when both sides spell it the same way: pushing down the
    canonical format_type must drop a layer's existing formatType, or the
    stale alias -- serialized after format_type in the wire dict -- would
    win instead."""
    chart = fm.regplot(_frame(), x=X("v", formatType="time"), y="w")
    keyword = fm.regplot(_frame(), x=X("v", format_type="number", format=".0%"), y="w").to_svg()
    post_build = chart.encode(x=X("v", format_type="number", format=".0%")).to_svg()
    assert post_build == keyword


def test_a_call_time_encode_kwarg_pushes_down_the_same_way_as_a_later_encode_call():
    """``encode={...}`` at the call lands post-build too, so it pushes down the same way."""
    plain = fm.regplot(_frame(), x="v", y="w").to_svg()
    keyword = fm.regplot(_frame(), x=X("v", scale=fm.LogScale()), y="w").to_svg()
    via_encode_kwarg = fm.regplot(
        _frame(), x="v", y="w", encode={"x": X("v", scale=fm.LogScale())}
    ).to_svg()
    assert via_encode_kwarg != plain
    assert via_encode_kwarg == keyword


@pytest.mark.parametrize("door", ["later_dot_encode", "call_time_encode_kwarg"])
def test_a_call_time_encode_kwarg_pushes_down_the_same_way_with_format_too(door):
    """The door-parametrized check above, repeated for format= (not just
    scale=): both post-build doors land a binding the same way, so a
    format= pushes down identically through either."""
    if door == "later_dot_encode":
        build = lambda: fm.regplot(_frame(), x="v", y="w").encode(x=X("v", format=".2f"))  # noqa: E731
    else:
        build = lambda: fm.regplot(_frame(), x="v", y="w", encode={"x": X("v", format=".2f")})  # noqa: E731
    plain = fm.regplot(_frame(), x="v", y="w").to_svg()
    keyword = fm.regplot(_frame(), x=X("v", format=".2f"), y="w").to_svg()
    styled = build().to_svg()
    assert styled != plain
    assert styled == keyword


@pytest.mark.parametrize("entry_fn,entry_name", [(fm.regplot, "regplot"), (fm.lmplot, "lmplot")])
def test_a_post_build_y_domain_or_log_scale_pushes_down_on_regplot_and_lmplot(entry_fn, entry_name):
    """5-3 binds x only; the same push-down applies to y too, on both
    regplot and lmplot (lmplot's own `hue=` sweep already covers its x;
    this is the first y-channel check for either)."""
    chart = entry_fn(_frame(), x="v", y="w")
    plain = chart.to_svg()
    keyword_log = entry_fn(_frame(), x="v", y=Y("w", scale=fm.LogScale())).to_svg()
    post_build_log = chart.encode(y=Y("w", scale=fm.LogScale())).to_svg()
    assert post_build_log != plain
    assert post_build_log == keyword_log

    keyword_domain = entry_fn(
        _frame(), x="v", y=Y("w", scale=fm.LinearScale(domain=(-5.0, 5.0)))
    ).to_svg()
    post_build_domain = chart.encode(y=Y("w", scale=fm.LinearScale(domain=(-5.0, 5.0)))).to_svg()
    assert post_build_domain == keyword_domain


def test_a_call_time_encode_kwarg_of_none_is_byte_identical_like_a_later_one():
    """``encode={'x': None}`` lands post-build the same way a later
    ``.encode(x=None)`` does: None is unset, not a rewrite."""
    plain = fm.regplot(_frame(), x="v", y="w").to_svg()
    via_encode_kwarg = fm.regplot(_frame(), x="v", y="w", encode={"x": None}).to_svg()
    assert via_encode_kwarg == plain


# ---------------------------------------------------------------------------
# A post-build option on another variable never leaks onto the figure's
# layers, whatever else the same encode() writes.
# ---------------------------------------------------------------------------


def test_a_post_build_option_on_another_variable_never_leaks_onto_the_pointed_to_channel():
    """A binding on another variable (`x="u"`) carries its own `scale=`
    into the record's `written` dict, but the record leaves that binding
    alone (`u` is not the called field), so its `scale=` must never reach
    the `v` axis the figure actually drew -- even when the SAME `.encode()`
    call also re-points y, which is where the leak used to come from."""
    frame = _frame()
    plain = fm.regplot(frame, x="v", y="w")
    title_only = plain.encode(y=Y("w", title="R")).to_svg()
    combined = plain.encode(x=X("u", scale=fm.LogScale()), y=Y("w", title="R")).to_svg()
    assert combined == title_only


def test_a_post_build_axis_replaces_the_calls_own_axis_on_the_built_layers():
    """Bug fix: the call's own `axis=` used to win over a later one (the
    layer already carried it, so the old "fill only if unset" merge never
    replaced it); the later write must win instead, as `Chart.encode`
    always replaces a channel's binding."""
    chart = fm.regplot(_frame(), x=X("v", axis=fm.Axis(label_angle=45)), y="w")
    angle_45 = chart.to_svg()
    angle_90_keyword = fm.regplot(_frame(), x=X("v", axis=fm.Axis(label_angle=90)), y="w").to_svg()
    angle_90_post_build = chart.encode(x=X("v", axis=fm.Axis(label_angle=90))).to_svg()
    assert angle_90_post_build != angle_45
    assert angle_90_post_build == angle_90_keyword


def test_residplot_encode_kwarg_on_x_is_honored_not_a_raw_unknown_column():
    """``residplot(encode={'x': X('v', title=...)})`` used to raise the
    misleading ``encoding 'x' references unknown column 'v'``, because the
    point layer's own encoding merged ``encode=``'s raw x straight in. The
    point layer now excludes x/y from that merge, so the chart-level
    record re-points the title through the usual door instead."""
    plain = fm.residplot(_frame(), x="v", y="w", lowess=True).to_svg()
    styled = fm.residplot(
        _frame(), x="v", y="w", lowess=True, encode={"x": X("v", title="T")}
    ).to_svg()
    assert styled != plain
    assert ">T<" in styled


def test_residplot_encode_kwarg_on_a_non_positional_channel_still_reaches_the_point_layer():
    chart = fm.residplot(_frame(), x="v", y="w", lowess=True)
    plain = chart.to_svg()
    colored = fm.residplot(
        _frame().with_columns(pl.Series("g", ["a", "a", "a", "a", "b", "b", "b", "b"])),
        x="v",
        y="w",
        lowess=True,
        encode={"color": "g"},
    ).to_svg()
    assert colored != plain


# ---------------------------------------------------------------------------
# title=/axis= keep reaching the axis on every door that lands an x/y
# binding, unaffected.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form_name", sorted(_EAGER_ENTRY_FORMS))
def test_title_and_axis_reach_the_axis(form_name):
    """title= and axis=None both push down byte-identical to the keyword
    door, on every eager form including the three no-scatter-drawn ones:
    the dedicated fit chart now carries a chart-level x/y encoding of its
    own (the same way every other layered statistic does), so the
    documented axis=None -- read from that chart-level channel, not a
    layer's -- hides the axis there too."""
    plain = _EAGER_ENTRY_FORMS[form_name]()
    titled = _EAGER_ENTRY_FORMS[form_name]().encode(x=X("v", title="Predictor"))
    keyword_titled = _build_with_keyword_binding(form_name, "x", X("v", title="Predictor"))
    assert titled.to_svg() != plain.to_svg()
    assert ">Predictor<" in titled.to_svg()
    assert titled.to_svg() == keyword_titled.to_svg()

    suppressed = _EAGER_ENTRY_FORMS[form_name]().encode(x=X("v", axis=None))
    keyword_suppressed = _build_with_keyword_binding(form_name, "x", X("v", axis=None))
    assert suppressed.to_svg() != plain.to_svg()
    assert suppressed.to_svg() == keyword_suppressed.to_svg()


def test_a_sort_on_the_called_field_is_inert_like_every_other_door():
    chart = fm.regplot(_frame(), x="v", y="w")
    plain = chart.to_svg()
    sorted_chart = chart.encode(x=X("v", sort="-x")).to_svg()
    assert plain == sorted_chart


# ---------------------------------------------------------------------------
# A field-less post-build channel on an INPUT axis is refused naming the
# called field to restate.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form_name", sorted(_EAGER_ENTRY_FORMS))
@pytest.mark.parametrize("channel,field,cls", [("x", "v", X)])
def test_a_fieldless_channel_on_an_input_axis_is_refused(form_name, channel, field, cls):
    build = _EAGER_ENTRY_FORMS[form_name]
    chart = build()
    entry = _entry_of(form_name)
    with pytest.raises(ValueError) as err:
        chart.encode(**{channel: cls(title="T")}).to_svg()
    message = str(err.value)
    assert message == (
        f"{entry}({channel}={cls.__name__}(title='T')) after the call is not supported: a "
        f"channel with no field replaces {channel}={field!r}, which the fit reads. Restate the "
        f"column: .encode({channel}=fm.{cls.__name__}({field!r}, title='T'))."
    )


def test_a_fieldless_y_on_regplot_is_refused_naming_y():
    chart = fm.regplot(_frame(), x="v", y="w")
    with pytest.raises(ValueError) as err:
        chart.encode(y=Y(title="T")).to_svg()
    assert str(err.value) == (
        "regplot(y=Y(title='T')) after the call is not supported: a channel with no field "
        "replaces y='w', which the fit reads. Restate the column: .encode(y=fm.Y('w', title='T'))."
    )


@pytest.mark.parametrize("channel,field,cls", [("x", "v", X), ("y", "w", Y)])
@pytest.mark.parametrize("door", ["later_dot_encode", "call_time_encode_kwarg"])
def test_a_pending_forms_fieldless_channel_is_refused_on_either_door(door, channel, field, cls):
    """5c-10/5c-11: a pending scatter=False smooth-routed form runs only
    the field-less rule from the record (its own resolve-time prep handles
    everything else), on both x and y, through both post-build doors."""
    plain = fm.regplot(_frame(), x="v", y="w", scatter=False)
    if door == "later_dot_encode":
        build = lambda: plain.encode(**{channel: cls(title="T")}).to_svg()  # noqa: E731
    else:
        build = lambda: fm.regplot(  # noqa: E731
            _frame(), x="v", y="w", scatter=False, encode={channel: cls(title="T")}
        ).to_svg()
    with pytest.raises(ValueError) as err:
        build()
    assert str(err.value) == (
        f"regplot({channel}={cls.__name__}(title='T')) after the call is not supported: a "
        f"channel with no field replaces {channel}={field!r}, which the fit reads. Restate the "
        f"column: .encode({channel}=fm.{cls.__name__}({field!r}, title='T'))."
    )


def test_a_pending_forms_fieldless_remedy_states_only_what_holds_on_the_form():
    """The pending form has no layers at all yet, so its fieldless remedy
    never claims built layers cannot read a blocked option. Its ending
    differs by declared type, not by a blanket pending-vs-eager split: a
    declared T restated there is dropped with no scatter drawn to apply
    it to, the same fact the dedicated no-scatter forms' own remedy
    states (in the caller's terms, not the pending mechanism's); a
    declared N/O restated there is refused again, by the pending
    preparation's own categorical rule, so it keeps the ordinary ending
    an eager form's identical binding genuinely gets."""
    pending = fm.regplot(_frame(), x="v", y="w", scatter=False)
    eager = fm.regplot(_frame(), x="v", y="w")
    with pytest.raises(ValueError) as pending_t_err:
        pending.encode(x=X(type="T", title="P")).to_svg()
    with pytest.raises(ValueError) as eager_err:
        eager.encode(x=X(type="T", title="P")).to_svg()
    assert str(pending_t_err.value) == (
        "regplot(x=X(type='T', title='P')) after the call is not supported: a channel with no "
        "field replaces x='v', which the fit reads. Restate the column, dropping type= "
        "(with no scatter drawn, a declared type is never applied): "
        ".encode(x=fm.X('v', title='P'))."
    )
    assert str(eager_err.value) == (
        "regplot(x=X(type='T', title='P')) after the call is not supported: a channel with no "
        "field replaces x='v', which the fit reads. Restate the column, dropping type= "
        "(restating with type= would be refused too): .encode(x=fm.X('v', title='P'))."
    )


@pytest.mark.parametrize("declared_type", ["N", "O"])
def test_a_pending_forms_fieldless_remedy_for_a_categorical_type_matches_the_eager_ending(
    declared_type,
):
    """A declared N/O restated on the pending form is refused again, by
    the pending preparation's own categorical rule -- unlike a declared T,
    it is never silently dropped there, so its remedy keeps the same
    ending an eager form's identical binding gets."""
    pending = fm.regplot(_frame(), x="v", y="w", scatter=False)
    eager = fm.regplot(_frame(), x="v", y="w")
    with pytest.raises(ValueError) as pending_err:
        pending.encode(x=X(type=declared_type, title="P")).to_svg()
    with pytest.raises(ValueError) as eager_err:
        eager.encode(x=X(type=declared_type, title="P")).to_svg()
    assert str(pending_err.value) == str(eager_err.value)
    assert str(pending_err.value) == (
        f"regplot(x=X(type={declared_type!r}, title='P')) after the call is not supported: a "
        f"channel with no field replaces x='v', which the fit reads. Restate the column, "
        f"dropping type= (restating with type= would be refused too): "
        f".encode(x=fm.X('v', title='P'))."
    )


def test_a_fieldless_scale_on_an_input_axis_is_refused_by_the_fieldless_rule_not_the_unreadable_one():
    chart = fm.regplot(_frame(), x="v", y="w")
    with pytest.raises(ValueError) as err:
        chart.encode(x=X(scale=fm.LogScale())).to_svg()
    # The fieldless template (below), not the unreadable-option one (which
    # would read "...would be ignored...").
    assert str(err.value) == (
        "regplot(x=X(scale=LogScale(...))) after the call is not supported: a channel with no "
        "field replaces x='v', which the fit reads. Restate the column: "
        ".encode(x=fm.X('v', scale=LogScale(...)))."
    )


def test_a_fieldless_binding_echoes_none_as_none_not_nonetype():
    """``fm.X(axis=None)`` quotes its written kwarg as ``axis=None``, not
    ``axis=NoneType(...)`` -- ``None`` is a plain literal to echo back, the
    same as any other value a caller actually wrote, not a pre-built
    object to name by class alone."""
    chart = fm.regplot(_frame(), x="v", y="w")
    with pytest.raises(ValueError) as err:
        chart.encode(x=X(axis=None)).to_svg()
    assert str(err.value) == (
        "regplot(x=X(axis=None)) after the call is not supported: a channel with no field "
        "replaces x='v', which the fit reads. Restate the column: .encode(x=fm.X('v', axis=None))."
    )


# ---------------------------------------------------------------------------
# residplot's y scale/domain/format refusal names only remedies that
# render -- a log scale never names the call door, since it refuses there
# too.
# ---------------------------------------------------------------------------


def test_residplot_y_log_scale_refusal_never_names_the_call_door():
    chart = fm.residplot(_frame(), x="v", y="w", lowess=True)
    with pytest.raises(ValueError) as err:
        chart.encode(y=Y("w", scale=fm.LogScale())).to_svg()
    message = str(err.value)
    assert message == (
        "residplot(y=Y('w', scale=LogScale(...))) "
        "after the call is not supported: residplot draws y from the y= it was called with, so "
        "scale= would be ignored. Use fm.SymlogScale(), or drop scale=."
    )
    assert "residplot(y=fm.Y(" not in message


def test_residplot_y_domain_scale_refusal_names_the_call_door():
    chart = fm.residplot(_frame(), x="v", y="w", lowess=True)
    with pytest.raises(ValueError) as err:
        chart.encode(y=Y("w", scale=fm.LinearScale(domain=(-5.0, 5.0)))).to_svg()
    assert str(err.value) == (
        "residplot(y=Y('w', scale=LinearScale(...))) after the call is not supported: residplot "
        "draws y from the y= it was called with, so scale= would be ignored. Drop scale=, or "
        "pass it in the call: residplot(y=fm.Y('w', scale=...))."
    )


def test_residplot_y_format_refusal_names_the_call_door():
    chart = fm.residplot(_frame(), x="v", y="w", lowess=True)
    with pytest.raises(ValueError) as err:
        chart.encode(y=Y("w", format=".2f")).to_svg()
    assert str(err.value) == (
        "residplot(y=Y('w', format='.2f')) after the call is not supported: residplot draws y "
        "from the y= it was called with, so format= would be ignored. Drop format=, or pass it "
        "in the call: residplot(y=fm.Y('w', format=...))."
    )


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (
            {"scale": fm.LogScale()},
            "residplot(y=Y('w', scale=LogScale(...))) after the call is not supported: residplot "
            "draws y from the y= it was called with, so scale= would be ignored. Use "
            "fm.SymlogScale(), or drop scale=.",
        ),
        (
            {"scale": fm.LinearScale(domain=(-5.0, 5.0))},
            "residplot(y=Y('w', scale=LinearScale(...))) after the call is not supported: "
            "residplot draws y from the y= it was called with, so scale= would be ignored. Drop "
            "scale=, or pass it in the call: residplot(y=fm.Y('w', scale=...)).",
        ),
        (
            {"format": ".2f"},
            "residplot(y=Y('w', format='.2f')) after the call is not supported: residplot draws "
            "y from the y= it was called with, so format= would be ignored. Drop format=, or "
            "pass it in the call: residplot(y=fm.Y('w', format=...)).",
        ),
    ],
)
def test_residplot_flat_y_scale_domain_or_format_refusal_matches_the_layered_form(kwargs, expected):
    """5-13: the layered form's y scale=/domain=/format= refusals above are
    pinned with lowess=True only; flat residplot (no overlay) goes through
    the exact same record, so it refuses identically."""
    flat = fm.residplot(_frame(), x="v", y="w", zero_line=False, show_metrics=False)
    layered = fm.residplot(_frame(), x="v", y="w", lowess=True)
    with pytest.raises(ValueError) as flat_err:
        flat.encode(y=Y("w", **kwargs)).to_svg()
    with pytest.raises(ValueError) as layered_err:
        layered.encode(y=Y("w", **kwargs)).to_svg()
    assert str(flat_err.value) == str(layered_err.value)
    assert str(flat_err.value) == expected


def test_residplot_y_styles_the_residual_axis_when_fieldless():
    chart = fm.residplot(_frame(), x="v", y="w", lowess=True)
    plain = chart.to_svg()
    styled = chart.encode(y=Y(title="Resid")).to_svg()
    assert styled != plain
    assert ">Resid<" in styled


def test_residplot_flat_y_styles_the_residual_axis_when_fieldless():
    chart = fm.residplot(_frame(), x="v", y="w", zero_line=False, show_metrics=False)
    plain = chart.to_svg()
    styled = chart.encode(y=Y(title="Resid")).to_svg()
    assert styled != plain
    assert ">Resid<" in styled


# ---------------------------------------------------------------------------
# A binding that names residplot's drawn columns directly stays honored,
# unaffected.
# ---------------------------------------------------------------------------


def test_residplot_drawn_columns_stay_honored():
    chart = fm.residplot(_frame(), x="v", y="w", lowess=True)
    plain = chart.to_svg()
    scaled_y = chart.encode(y=Y("residual", scale=fm.LinearScale(domain=(-5.0, 5.0)))).to_svg()
    formatted_x = chart.encode(x=X("x", format=".2f")).to_svg()
    assert scaled_y != plain
    assert formatted_x != plain


def test_residplot_drawn_column_y_log_scale_is_still_rust_refused():
    chart = fm.residplot(_frame(), x="v", y="w", lowess=True)
    with pytest.raises(ValueError) as err:
        chart.encode(y=Y("residual", scale=fm.LogScale())).to_svg()
    assert str(err.value) == (
        "residplot(y=<log scale>) is not supported: the residual axis draws signed residuals "
        "around 0, which a LogScale cannot place. Use fm.SymlogScale(), or drop scale="
    )


@pytest.mark.parametrize("lowess", [False, True])
def test_residplot_predictor_named_like_the_drawn_column_stays_honored(lowess):
    """residplot's x draws the predictor unchanged, so when the caller's own
    predictor happens to be named 'x' (the same column name residplot draws
    it under), a post-build binding naming it is a second spelling of the
    figure's own drawn-column binding, read as that is -- not refused as a
    rewrite of the called field, even though the field also equals the
    called field here."""
    frame = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "y": [2.0, 3.0, 1.0, 5.0, 4.0, 2.0, 6.0, 3.0],
        }
    )
    kwargs = {"lowess": True} if lowess else {"zero_line": False, "show_metrics": False}
    chart = fm.residplot(frame, x="x", y="y", **kwargs)
    plain = chart.to_svg()
    styled = chart.encode(x=X("x", format=".2f")).to_svg()
    assert styled != plain


def test_residplot_response_named_residual_is_the_called_field_after_the_call():
    """The caller's data is never renamed: a response column literally
    named 'residual' stays 'residual', and the COMPUTED residual takes a
    fresh name instead. So 'residual' is the
    CALLED field here, not the drawn column: a later binding naming it
    re-points like any other called-field binding -- a title restates and
    reaches the axis -- but scale=/format= are still refused, the same
    way they always are on residplot's y, since the response column the
    call named is not the residual the figure draws."""
    frame = pl.DataFrame(
        {
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "residual": [2.0, 3.0, 1.0, 5.0, 4.0, 2.0, 6.0, 3.0],
        }
    )
    chart = fm.residplot(frame, x="v", y="residual", zero_line=False, show_metrics=False)
    plain = chart.to_svg()
    styled = chart.encode(y=Y("residual", title="T")).to_svg()
    assert styled != plain
    assert ">T<" in styled
    with pytest.raises(ValueError) as err:
        chart.encode(y=Y("residual", format=".2f")).to_svg()
    assert str(err.value) == (
        "residplot(y=Y('residual', format='.2f')) after the call is not supported: "
        "residplot draws y from the y= it was called with, so format= would be "
        "ignored. Drop format=, or pass it in the call: "
        "residplot(y=fm.Y('residual', format=...))."
    )


@pytest.mark.parametrize("declared_type", ["N", "O", "T"])
def test_residplot_fieldless_y_with_a_disallowed_declared_type_is_refused(declared_type):
    """A field-less y (which styles the computed residual axis instead of
    being refused) still refuses a declared N/O/T: the residual axis is
    never a category or a time, whether the caller wrote the type on a
    field-bearing or a field-less binding."""
    chart = fm.residplot(_frame(), x="v", y="w", zero_line=False, show_metrics=False)
    with pytest.raises(ValueError) as err:
        chart.encode(y=Y(type=declared_type, title="R")).to_svg()
    placement = "timestamps" if declared_type == "T" else "categories"
    assert str(err.value) == (
        f"residplot(y=Y(type={declared_type!r})) is not supported: the residual axis draws "
        f"numbers around 0, so type={declared_type!r} would place the residuals as {placement}. "
        "Drop the type (y='residual')."
    )


def test_residplot_flat_post_build_title_only_keeps_a_declared_temporal_x():
    """Flat residplot draws x straight from the chart's own encoding, with
    no layer of its own to shield a call-time declared T from a later
    binding that re-points it with no type of its own
    (`_rebind_positional` forces every re-point to Q): the record restores
    the call-time declared type after the force, the same restoration the
    call-time door itself runs."""
    frame = pl.DataFrame(
        {
            "v": [1.6e12, 1.61e12, 1.62e12, 1.63e12, 1.64e12, 1.65e12, 1.66e12, 1.67e12],
            "w": [2.0, 3.0, 1.0, 5.0, 4.0, 2.0, 6.0, 3.0],
        }
    )
    chart = fm.residplot(frame, x="v:T", y="w", zero_line=False, show_metrics=False)
    baseline = chart.to_svg()
    styled = chart.encode(x=X("v", title="P")).to_svg()
    assert styled != baseline
    assert ">P<" in styled
    # The temporal tick format survives (e.g. "Oct 2021"), not a raw epoch
    # number like "1.6e12" -- the same temporal axis the call-time door draws.
    from tests._snapshots import legend_texts

    assert any("e12" in text for text in legend_texts(baseline)) is False
    assert any("e12" in text for text in legend_texts(styled)) is False
    assert any(
        text.strip().endswith("2021") or text.strip().endswith("2022")
        for text in legend_texts(styled)
    )


def test_residplot_flat_post_build_type_differing_from_a_declared_temporal_call_is_refused():
    """The sibling case to the test above: a post-build type that DIFFERS
    from the call's own declared T is refused, naming the call door, not
    silently applied over it. `_rebind_positional` forces every re-point
    to Q and `restore_declared_type` restores the call's T afterward, but
    only once a differing declared type has already been refused above --
    otherwise a post-build `type='Q'` would silently overwrite the
    caller's explicit `'v:T'` with no layer of its own to shield it (flat
    residplot reads x straight off the chart-level encoding)."""
    frame = pl.DataFrame(
        {
            "v": [1.6e12, 1.61e12, 1.62e12, 1.63e12, 1.64e12, 1.65e12, 1.66e12, 1.67e12],
            "w": [2.0, 3.0, 1.0, 5.0, 4.0, 2.0, 6.0, 3.0],
        }
    )
    chart = fm.residplot(frame, x="v:T", y="w", zero_line=False, show_metrics=False)
    with pytest.raises(ValueError) as err:
        chart.encode(x=X("v", type="Q", title="P")).to_svg()
    assert str(err.value) == (
        "residplot(x=X('v', type='Q')) after the call is not supported: residplot draws x from "
        "the x= it was called with, so type= would be ignored. Drop type=, or pass it in the "
        "call: residplot(x=fm.X('v', type=...))."
    )


# ---------------------------------------------------------------------------
# A mark call that replaces a figure's mark clears the figure's post-build
# record too, the same way it clears _composite_source/_composite_kind.
# ---------------------------------------------------------------------------


def test_a_mark_call_after_a_pending_composite_speaks_the_marks_own_refusal():
    chart = fm.regplot(_frame(), x="v", y="w", scatter=False)
    with pytest.raises(ValueError) as err:
        chart.mark_point().encode(x=X(title="T")).to_svg()
    # mark_point's own fieldless-input refusal, not regplot's re-voiced one.
    assert str(err.value) == (
        "mark_point needs both positional channels, but nothing binds 'x'. Encode 'x' on the "
        "chart to draw it, or draw from 'y' alone with mark_tick() or mark_rule()."
    )


def test_a_mark_call_after_flat_residplot_speaks_the_marks_own_refusal():
    chart = fm.residplot(_frame(), x="v", y="w", zero_line=False, show_metrics=False)
    with pytest.raises(ValueError) as err:
        chart.mark_line().encode(x=X(title="T")).to_svg()
    # mark_line's own fieldless-input refusal, not residplot's re-voiced one.
    assert str(err.value) == (
        "mark_line needs both positional channels, but nothing binds 'x'. Encode 'x' on the "
        "chart to draw it, or draw from 'y' alone with mark_tick() or mark_rule()."
    )


# ---------------------------------------------------------------------------
# A pending smooth-routed form that `_finalize_chart`'s own `mark=` later
# resolves into an eager one is read against the chart it actually is,
# decided live when the record is applied -- not the shape it was in
# when the record was built.
# ---------------------------------------------------------------------------


def test_a_pending_form_resolved_eager_by_mark_reads_the_record_as_eager():
    """``mark=`` resolves the pending form at the call, into a chart whose
    layer the record's push-down reaches -- so format= cannot
    discriminate this chart from a genuinely pending one (both honor it);
    a declared type still tells them apart, since this chart's
    `scatter=False` keeps dropping one with no scatter drawn, while a
    genuinely pending chart defers the whole question to its own
    resolve-time prep (the test below)."""
    chart = fm.regplot(_frame(), x="v", y="w", scatter=False, mark={"line": {"stroke": "red"}})
    with pytest.raises(ValueError) as err:
        chart.encode(x=X("v", type="T")).to_svg()
    assert str(err.value) == (
        "regplot(x=X('v', type='T')) after the call is not supported: regplot draws x from the "
        "x= it was called with, so type= would be ignored. Drop type=: with no scatter drawn, a "
        "declared type is never applied."
    )


@pytest.mark.parametrize("door", ["keyword", "call_time_encode_kwarg", "later_dot_encode"])
def test_a_genuinely_pending_form_honors_format_on_every_door(door):
    """A pending scatter=False smooth-routed regplot (the default method,
    no mark= override) builds no layers at the call: its own resolve-time
    preparation reads a field-bearing format= from the final encoding
    directly, on every door that lands one, not just the keyword door.
    `Chart._resolve_pending` runs more than once reaching one render (once
    directly in `_render_inputs`, again inside `to_spec`), so this also
    pins that the second, redundant run does not re-refuse a binding the
    first run already handed to the pending prep."""
    frame = _frame()
    baseline = fm.regplot(frame, x="v", y="w", scatter=False).to_svg()
    if door == "keyword":
        styled = fm.regplot(frame, x=X("v", format=".3f"), y="w", scatter=False).to_svg()
    elif door == "call_time_encode_kwarg":
        styled = fm.regplot(
            frame, x="v", y="w", scatter=False, encode={"x": X("v", format=".3f")}
        ).to_svg()
    else:
        styled = (
            fm.regplot(frame, x="v", y="w", scatter=False).encode(x=X("v", format=".3f")).to_svg()
        )
    assert styled != baseline


def test_a_genuinely_pending_forms_declared_type_is_not_refused_by_the_eager_record():
    """The discriminator in the mark-override test above: a genuinely
    pending chart defers a declared type to its own resolve-time prep
    (not the eager record), so it does not raise here, unlike the
    mark-resolved-eager chart."""
    chart = fm.regplot(_frame(), x="v", y="w", scatter=False)
    chart.encode(x=X("v", type="T")).to_svg()


# ---------------------------------------------------------------------------
# The statistic refusal (bin=/aggregate=) runs before the field-less rule
# on every form, pending included -- a caller's bare bin=/aggregate=
# always gets this text, never the pending prep's own, differently-worded
# refusal moments later.
# ---------------------------------------------------------------------------


def test_pending_forms_aggregate_refusal_runs_before_the_fieldless_rule():
    chart = fm.regplot(_frame(), x="v", y="w", scatter=False)
    with pytest.raises(ValueError) as err:
        chart.encode(y=Y(aggregate="count")).to_svg()
    assert str(err.value) == (
        "regplot(y=Y(aggregate='count')) is not supported: the fit reads the raw response, so "
        "aggregate= would be ignored. Drop aggregate=."
    )


# ---------------------------------------------------------------------------
# A binding on another variable (not the called field, and not a drawn
# column) is left alone, unaffected.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form_name", sorted(set(_EAGER_ENTRY_FORMS) - {"residplot_flat"}))
def test_a_post_build_binding_on_another_variable_keeps_its_current_output(form_name):
    build = _EAGER_ENTRY_FORMS[form_name]
    plain = build()
    rebound = build().encode(x="u")
    assert plain.to_svg() == rebound.to_svg()


#: Each another-variable door on flat residplot and the exact refusal it
#: raises: a later `.encode()`, and `encode=` at the call (which lands on
#: the chart the same way, after the layers are already built).
_FLAT_ANOTHER_VARIABLE_DOORS: dict[str, tuple[Callable[[], "fm.Chart"], str]] = {
    "later-x": (
        lambda: _residplot_flat().encode(x="u"),
        "residplot(x=X('u')) after the call is not supported: residplot computes its "
        "residuals from the x= it was called with ('v'), so 'u' would be drawn without "
        "a refit. Pass it in the call instead: residplot(x='u', ...).",
    ),
    "later-y": (
        lambda: _residplot_flat().encode(y="u"),
        "residplot(y=Y('u')) after the call is not supported: residplot computes its "
        "residuals from the y= it was called with ('w'), so 'u' would be drawn without "
        "a refit. Pass it in the call instead: residplot(y='u', ...).",
    ),
    "encode-x": (
        lambda: fm.residplot(
            _frame(), x="v", y="w", zero_line=False, show_metrics=False, encode={"x": "u"}
        ),
        "residplot(x=X('u')) after the call is not supported: residplot computes its "
        "residuals from the x= it was called with ('v'), so 'u' would be drawn without "
        "a refit. Pass it in the call instead: residplot(x='u', ...).",
    ),
    "encode-y": (
        lambda: fm.residplot(
            _frame(), x="v", y="w", zero_line=False, show_metrics=False, encode={"y": "u"}
        ),
        "residplot(y=Y('u')) after the call is not supported: residplot computes its "
        "residuals from the y= it was called with ('w'), so 'u' would be drawn without "
        "a refit. Pass it in the call instead: residplot(y='u', ...).",
    ),
    "later-x-named-x": (
        # The data's own 'x' column is not the predictor ('v' is), so it is
        # another variable -- even though its literal name matches what
        # residplot calls the predictor's drawn column internally. When the
        # data holds its own 'x', the computed one takes a fresh name
        # instead (the naming mechanism's own rule), so this 'x' is
        # genuinely the caller's column, never residplot's drawn one.
        lambda: fm.residplot(
            _frame().with_columns(pl.col("u").alias("x")),
            x="v",
            y="w",
            zero_line=False,
            show_metrics=False,
        ).encode(x="x"),
        "residplot(x=X('x')) after the call is not supported: residplot computes its "
        "residuals from the x= it was called with ('v'), so 'x' would be drawn without "
        "a refit. Pass it in the call instead: residplot(x='x', ...).",
    ),
    "later-y-named-residual": (
        # Symmetric case on y: the data's own 'residual' is not the
        # response ('w' is), so it is another variable, not the computed
        # residual (which took a fresh name once this clash existed).
        lambda: fm.residplot(
            _frame().with_columns(pl.col("u").alias("residual")),
            x="v",
            y="w",
            zero_line=False,
            show_metrics=False,
        ).encode(y="residual"),
        "residplot(y=Y('residual')) after the call is not supported: residplot computes its "
        "residuals from the y= it was called with ('w'), so 'residual' would be drawn without "
        "a refit. Pass it in the call instead: residplot(y='residual', ...).",
    ),
}


@pytest.mark.parametrize("door", sorted(_FLAT_ANOTHER_VARIABLE_DOORS))
def test_a_post_build_binding_on_another_variable_is_refused_on_flat_residplot(door):
    """Flat residplot's one mark reads x/y straight from the chart-level
    encoding, with no layer of its own to keep drawing the fit's residuals
    once that binding points somewhere else: naming another column of the
    data -- 'u', neither the called field nor a drawn column -- is refused
    naming the call as the remedy, rather than silently drawing that
    column's raw values (or the fit's residuals at its positions) under
    the residual axis's title. An EntryRefusal, re-voiceable as any public
    call that reaches flat residplot's builder, not a bare ValueError."""
    build, text = _FLAT_ANOTHER_VARIABLE_DOORS[door]
    with pytest.raises(EntryRefusal) as err:
        build().to_svg()
    assert str(err.value) == text


def test_a_post_build_binding_on_a_column_the_data_lacks_keeps_flat_residplots_raw_failure():
    """'zz' names no column at all: Python cannot carry a column that does
    not exist, so this stays the same raw failure it always was, not the
    another-variable refusal (which would misname 'zz' as a column the
    fit could have refit on)."""
    with pytest.raises(ValueError) as err:
        _residplot_flat().encode(x="zz").to_svg()
    assert str(err.value) == "unknown column 'zz' referenced by an encoding"


def test_a_post_build_binding_on_the_called_field_re_points_onto_a_fresh_computed_column():
    """The called field ('v') still re-points through a later binding even
    when the data's own clashing 'x' column pushed the computed x onto a
    fresh name ('x_1', not the literal 'x'): the re-point has to use
    whatever fresh name this call actually chose, not a hardcoded 'x'."""
    clash = _frame().with_columns(pl.col("u").alias("x"))
    chart = fm.residplot(clash, x="v", y="w", zero_line=False, show_metrics=False)
    plain = chart.to_svg()
    styled = chart.encode(x=X("v", format=".2f")).to_svg()
    assert styled != plain
    # A sibling with no clash at all (so the computed x stays the literal
    # 'x', never renamed) renders byte-identically under the same style:
    # proof the re-point landed on the SAME predictor values, through
    # whichever fresh name this call's own naming chose, not a copy the
    # clashing 'x' column itself supplied.
    no_clash = _frame()
    sibling = fm.residplot(no_clash, x="v", y="w", zero_line=False, show_metrics=False)
    sibling_styled = sibling.encode(x=X("v", format=".2f")).to_svg()
    assert styled == sibling_styled


# ---------------------------------------------------------------------------
# `+` refuses what the left operand would refuse alone; the composition's
# own binding, written after the `+`, is the composition's (the record
# resets on the `+` result, as `_composite_source` does).
# ---------------------------------------------------------------------------


def test_a_post_build_refusal_fires_at_plus_too():
    left = fm.regplot(_frame(), x="v", y="w").encode(x=X("v", bin=True))
    right = fm.Chart(_frame()).mark_point().encode(x="v", y="w")
    with pytest.raises(ValueError) as err:
        (left + right).to_svg()
    assert str(err.value) == (
        "regplot(x=X('v', bin=True)) is not supported: the fit reads the raw predictor, so bin= "
        "would be ignored. Bin the predictor with x_bins=<number of bins>, the fit's own "
        "binning, or drop bin=."
    )


def test_a_composition_binding_after_plus_is_the_compositions_own():
    left = fm.regplot(_frame(), x="v", y="w")
    right = fm.Chart(_frame()).mark_point().encode(x="v", y="w")
    composed = left + right
    # The composition's own later binding is not run through regplot's
    # post-build record at all -- the same as any binding on an
    # unrelated field above -- so it simply renders.
    composed.encode(x=X("v", bin=True)).to_svg()


# ---------------------------------------------------------------------------
# Order of refusals: the statistic refusal (bin=/aggregate=) runs before
# the categorical rule, which runs before the unreadable-option refusal.
# ---------------------------------------------------------------------------


def test_bin_and_scale_together_are_refused_as_a_bin_first():
    chart = fm.regplot(_frame(), x="v", y="w")
    with pytest.raises(ValueError) as err:
        chart.encode(x=X("v", bin=True, scale=fm.LogScale())).to_svg()
    assert str(err.value) == (
        "regplot(x=X('v', bin=True)) is not supported: the fit reads the raw predictor, so bin= "
        "would be ignored. Bin the predictor with x_bins=<number of bins>, the fit's own "
        "binning, or drop bin=."
    )


def test_a_categorical_type_and_scale_together_are_refused_as_categorical_first():
    chart = fm.regplot(_frame(), x="v", y="w")
    with pytest.raises(ValueError) as err:
        chart.encode(x=X("v", type="N", scale=fm.LogScale())).to_svg()
    assert str(err.value) == (
        "regplot(x=X('v', type='N')) is not supported: the fit reads x as numbers and draws its "
        "curve along a continuous axis, so type='N' would place the fitted values as categories. "
        "Drop the type (x='v')."
    )


# ---------------------------------------------------------------------------
# None is unset; a wrong value type raises the channel's own TypeError.
# ---------------------------------------------------------------------------


def test_post_build_x_none_leaves_the_binding_as_is():
    chart = fm.regplot(_frame(), x="v", y="w")
    plain = chart.to_svg()
    assert chart.encode(x=None).to_svg() == plain
    assert chart.encode(**{"x": None}).to_svg() == plain


def test_unset_scale_and_format_on_a_post_build_channel_are_byte_identical():
    chart = fm.regplot(_frame(), x="v", y="w")
    plain = chart.to_svg()
    unset = chart.encode(x=X("v", scale=None, format=None)).to_svg()
    assert unset == plain


def test_unset_format_on_a_layered_statistic_is_byte_identical():
    """``format=None`` on a layered statistic (not a regression figure) is
    unset, the same as on any other door -- no option to drop, nothing to
    propagate."""
    chart = fm.Chart(_frame()).mark_smooth(ci=0.95)
    plain = chart.encode(x="v", y="w").to_svg()
    unset = chart.encode(x=X("v", format=None), y="w").to_svg()
    assert unset == plain


def test_a_wrong_type_channel_value_still_raises_the_existing_type_error():
    chart = fm.regplot(_frame(), x="v", y="w")
    with pytest.raises(TypeError):
        chart.encode(x=5)


@pytest.mark.parametrize("kwargs,exc", [({"format": 5}, TypeError), ({"scale": 5}, ValueError)])
def test_a_wrong_type_unreadable_option_raises_the_channels_own_validation_not_the_unreadable_refusal(
    kwargs, exc
):
    """``format=5``/``scale=5`` never had a valid shape in the first place
    (they fail the same way on a plain chart), so the post-build door
    leaves them alone rather than refusing them as unreadable with a
    remedy that would just fail again for an unrelated reason."""
    chart = fm.regplot(_frame(), x="v", y="w")
    plain_chart = fm.Chart(_frame()).mark_point().encode(x="v", y="w")
    with pytest.raises(exc) as post_build_err:
        chart.encode(x=X("v", **kwargs)).to_svg()
    with pytest.raises(exc) as plain_err:
        plain_chart.encode(x=X("v", **kwargs)).to_svg()
    assert str(post_build_err.value) == str(plain_err.value)


# ---------------------------------------------------------------------------
# Degenerate data: refusals fire identically on zero-row and one-row data,
# since they read bindings, not rows.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("height", [0, 1])
def test_a_post_build_refusal_fires_identically_on_degenerate_data(height):
    frame = _frame().head(height)
    chart = fm.regplot(frame, x="v", y="w")
    with pytest.raises(ValueError) as err:
        chart.encode(x=X("v", bin=True)).to_svg()
    assert str(err.value) == (
        "regplot(x=X('v', bin=True)) is not supported: the fit reads the raw predictor, so bin= "
        "would be ignored. Bin the predictor with x_bins=<number of bins>, the fit's own "
        "binning, or drop bin=."
    )


@pytest.mark.parametrize("height", [0, 1])
def test_a_post_build_honored_option_renders_with_the_degenerate_fit_warning(height):
    """An honored option (title=) on degenerate data still renders -- with
    the usual degenerate-fit warning (too few points for Smooth to fit), not
    a refusal: the record's own gatekeeping is about which OPTIONS reach
    the layers, not about how many rows there are to fit."""
    frame = _frame().head(height)
    chart = fm.regplot(frame, x="v", y="w")
    with pytest.warns(UserWarning, match="could not fit the data"):
        svg = chart.encode(x=X("v", title="Predictor")).to_svg()
    assert ">Predictor<" in svg


# ---------------------------------------------------------------------------
# Sweep: every fm.X/fm.Y accept, on the keyword door and the post-build
# door alike, lands on one of three known outcomes -- refused, changes
# the output, or is byte-identical (inert) -- never silently dropped. The
# two doors agree for every classified option here: scale=/format=/
# format_type=/axis= push down onto the figure's own built layers the
# same way the keyword door bakes them in at construction, so the only
# remaining divergence between a post-build write and the matching
# keyword write is residplot's y, which the multi-form sweep below checks
# separately (its own refusal is pinned
# directly by the dedicated residplot tests above). _CHANNEL_OPTIONS,
# derived from each option's measured behavior, is the record this test
# checks against -- a new fm.X/fm.Y accept with no entry here fails
# loudly rather than silently passing as "inert" (see the population
# guard right below the table).
# ---------------------------------------------------------------------------


#: One representative value per :attr:`ChannelBase._honored_kwargs` key, and
#: its outcome on the keyword door (baked into the layers at construction)
#: and the post-build door (``Chart._resolve_pending``'s record, pushed
#: down onto the already-built layers).
_CHANNEL_OPTIONS: dict[str, tuple[dict, str, str]] = {
    "bin": ({"bin": True}, "refused", "refused"),
    "aggregate": ({"aggregate": "mean"}, "refused", "refused"),
    "scale": ({"scale": fm.LogScale()}, "differs", "differs"),
    "format": ({"format": ".2f"}, "differs", "differs"),
    # "time" (not the default "number") so a drop and an honor read apart:
    # it is refused by Rust's own scale/format-type mismatch on BOTH doors
    # once honored, the same text either way -- not a silent post-build drop.
    "format_type": ({"format_type": "time"}, "refused", "refused"),
    # formatType is format_type's camelCase (Vega-compat) alias -- its own
    # entry, not folded into format_type's, so a push-down that reads one
    # spelling and not the other fails here instead of passing silently.
    "formatType": ({"formatType": "time"}, "refused", "refused"),
    "type": ({"type": "N"}, "refused", "refused"),
    "title": ({"title": "T"}, "differs", "differs"),
    # fm.Axis(...) (not the no-op None) so a drop and an honor read apart too.
    "axis": ({"axis": fm.Axis(label_angle=45)}, "differs", "differs"),
    "sort": ({"sort": "-x"}, "identical", "identical"),
    "impute": ({"impute": "zero"}, "identical", "identical"),
    "stack": ({"stack": True}, "identical", "identical"),
    "legend": ({"legend": None}, "identical", "identical"),
}


def test_channel_options_classifies_every_honored_kwarg():
    """A new fm.X/fm.Y accept with no entry in :data:`_CHANNEL_OPTIONS`
    must fail here first, rather than silently reading as untested by
    the sweeps below."""
    assert set(_CHANNEL_OPTIONS) == X._honored_kwargs
    assert X._honored_kwargs == Y._honored_kwargs


def _outcome(build, baseline: str) -> str:
    try:
        svg = build()
    except ValueError:
        return "refused"
    return "identical" if svg == baseline else "differs"


@pytest.mark.parametrize("option", sorted(_CHANNEL_OPTIONS))
def test_every_channel_option_is_honored_refused_or_inert_on_every_door(option):
    kwargs, keyword_expected, post_build_expected = _CHANNEL_OPTIONS[option]
    plain = fm.regplot(_frame(), x="v", y="w")
    baseline = plain.to_svg()

    keyword_outcome = _outcome(
        lambda: fm.regplot(_frame(), x=X("v", **kwargs), y="w").to_svg(), baseline
    )
    post_build_outcome = _outcome(lambda: plain.encode(x=X("v", **kwargs)).to_svg(), baseline)

    assert keyword_outcome == keyword_expected, (option, "keyword")
    assert post_build_outcome == post_build_expected, (option, "post-build")


@pytest.mark.parametrize("door", ["later_dot_encode", "call_time_encode_kwarg"])
@pytest.mark.parametrize("channel", ["x", "y"])
@pytest.mark.parametrize("form_name", sorted(_FULL_SWEEP_FORMS))
@pytest.mark.parametrize("option", sorted(_CHANNEL_OPTIONS))
def test_every_channel_option_is_honored_refused_or_inert_on_every_form_and_door(
    form_name, option, channel, door
):
    """The sweep above, run over every form's x AND y (every eager form,
    plus a genuinely pending scatter=False regplot and an lmplot(col=)
    facet grid), through both post-build doors (``encode=`` at the call,
    and a later ``.encode()``) -- not just regplot's x through
    ``.encode()`` alone. Both doors land a binding the same way (``encode=``
    is applied before ``_resolve_pending`` runs, the same record a later
    ``.encode()`` is read against), so they share one expected outcome per
    option; the keyword door's own outcome is checked once, on regplot's x,
    by the test above. residplot's y is the one exception for scale=/format=:
    it keeps refusing them, since the response column the call named is not
    the residual the figure draws."""
    kwargs, _keyword_expected, post_build_expected = _CHANNEL_OPTIONS[option]
    if option in ("scale", "format") and _entry_of(form_name) == "residplot" and channel == "y":
        post_build_expected = "refused"
    build = _FULL_SWEEP_FORMS[form_name]
    plain = build()
    baseline = plain.to_svg()
    field = "v" if channel == "x" else _y_field_of(form_name)
    cls = X if channel == "x" else Y
    binding = cls(field, **kwargs)
    if door == "later_dot_encode":
        outcome = _outcome(lambda: plain.encode(**{channel: binding}).to_svg(), baseline)
    else:
        outcome = _outcome(
            lambda: _build_with_encode_kwarg(form_name, channel, binding).to_svg(), baseline
        )
    assert outcome == post_build_expected, (form_name, option, channel, door)
