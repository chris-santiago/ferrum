"""``residplot`` refuses an ``x2=``/``y2=`` binding, which it draws on no layer.

A residual scatter draws each residual as one point, and its zero rule,
corner metrics and lowess overlay are per-fit summaries the caller never
positioned. A binding on either secondary positional channel can only bend
a summary layer or stretch an axis, so the figure refuses it on every door
that writes one -- the keyword, ``encode=``, and a later ``.encode()``
before or after ``+`` -- naming the drop as the remedy. ``None`` leaves the
channel unset, a value that is no binding keeps ``Chart.encode``'s own
``TypeError``, and a binding the caller writes on a composition or on a
chart whose mark they replaced is theirs, not the figure's.
"""

from __future__ import annotations

import pickle
import warnings

import polars as pl
import pytest

import ferrum as fm
from ferrum._entry import EntryRefusal

_FLAT = {"zero_line": False, "show_metrics": False}

#: Every form of the figure: the default layered one, the flat one with no
#: overlay at all, and the two alternative residual fits.
_FORMS: dict[str, dict] = {
    "layered": {},
    "flat": _FLAT,
    "lowess": {"lowess": True},
    "robust": {"robust": True},
}

_CHANNEL_CLASS = {"x2": fm.X2, "y2": fm.Y2}


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "w": [2.0, 3.0, 1.0, 5.0, 4.0, 2.0, 6.0, 3.0],
            "u": [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0],
        }
    )


def _refused_text(channel: str, written: str) -> str:
    """The sentence the figure refuses *written* on *channel* with."""
    return (
        f"residplot({channel}={written}) is not supported: residplot draws each "
        f"residual as a point, which has no {channel}. Drop {channel}=."
    )


def _residplot(form: str, **encoded: object) -> "fm.Chart":
    return fm.residplot(_frame(), x="v", y="w", **_FORMS[form], **encoded)


def _write(channel: str, value: object, door: str, form: str) -> "fm.Chart":
    """The figure with *value* written on *channel* through *door*."""
    if door == "keyword":
        return _residplot(form, **{channel: value})
    if door == "encode":
        return _residplot(form, encode={channel: value})
    assert door == "later"
    return _residplot(form).encode(**{channel: value})


@pytest.mark.parametrize("form", sorted(_FORMS))
@pytest.mark.parametrize("channel", sorted(_CHANNEL_CLASS))
@pytest.mark.parametrize("door", ["keyword", "encode", "later"])
@pytest.mark.parametrize("field", ["u", "v"])
def test_residplot_refuses_an_undrawn_channel_on_every_door(form, channel, door, field):
    expected = _refused_text(channel, f"{_CHANNEL_CLASS[channel].__name__}({field!r})")
    with pytest.raises(EntryRefusal) as caught:
        _write(channel, field, door, form).to_svg()
    assert str(caught.value) == expected


@pytest.mark.parametrize("form", sorted(_FORMS))
def test_a_missing_column_is_refused_as_undrawn(form):
    expected = _refused_text("x2", "X2('nope')")
    with pytest.raises(EntryRefusal) as caught:
        _residplot(form, x2="nope").to_svg()
    assert str(caught.value) == expected
    with pytest.raises(EntryRefusal) as caught:
        _residplot(form).encode(x2="nope").to_svg()
    assert str(caught.value) == expected


@pytest.mark.parametrize(
    ("binding", "channel", "written"),
    [
        (fm.X2("u"), "x2", "X2('u')"),
        (fm.Y2("u", aggregate="mean"), "y2", "Y2('u', aggregate='mean')"),
        (fm.X2(), "x2", "X2()"),
        (fm.X2(aggregate="mean"), "x2", "X2(aggregate='mean')"),
    ],
)
def test_the_refusal_spells_the_binding_as_written(binding, channel, written):
    expected = _refused_text(channel, written)
    with pytest.raises(EntryRefusal) as caught:
        _residplot("layered", **{channel: binding})
    assert str(caught.value) == expected
    with pytest.raises(EntryRefusal) as caught:
        _residplot("layered").encode(**{channel: binding}).to_svg()
    assert str(caught.value) == expected


@pytest.mark.parametrize("channel", ["x2", "y2"])
@pytest.mark.parametrize("door", ["keyword", "encode", "later"])
def test_a_value_that_is_no_binding_keeps_its_type_error(channel, door):
    expected = (
        f"encode({channel}=...) expects str, {_CHANNEL_CLASS[channel].__name__} "
        "instance, or Repeat placeholder; got int"
    )
    with pytest.raises(TypeError) as caught:
        _write(channel, 5, door, "layered")
    assert str(caught.value) == expected


@pytest.mark.parametrize("form", ["layered", "flat"])
@pytest.mark.parametrize("channel", ["x2", "y2"])
@pytest.mark.parametrize("door", ["keyword", "encode", "later"])
def test_none_leaves_the_channel_unset(form, channel, door):
    plain = fm.residplot(_frame(), x="v", y="w", **_FORMS[form]).to_svg()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _write(channel, None, door, form).to_svg() == plain
    assert caught == []


def test_a_binding_written_before_plus_is_refused():
    with pytest.raises(EntryRefusal) as caught:
        _residplot("layered").encode(x2="u") + fm.Chart(_frame()).mark_point().encode(x="v", y="w")
    assert str(caught.value) == _refused_text("x2", "X2('u')")


@pytest.mark.parametrize(
    ("channel", "written"),
    [("x2", "X2('u')"), ("y2", "Y2('u')")],
)
def test_residplot_on_the_right_of_plus_refuses_a_binding(channel, written):
    """A residplot operand refuses its own later binding when ``+`` resolves it.

    ``+`` resolves each operand's post-build record before it composes them,
    so the refusal fires there: no composition is built and nothing renders.
    """
    other = fm.Chart(_frame()).mark_point().encode(x="v", y="w")
    expected = _refused_text(channel, written)
    with pytest.raises(EntryRefusal) as caught:
        (other + _residplot("layered").encode(**{channel: "u"})).to_svg()
    assert str(caught.value) == expected


def test_a_later_binding_is_refused_at_to_spec_too():
    with pytest.raises(EntryRefusal) as caught:
        _residplot("layered").encode(y2="u").to_spec()
    assert str(caught.value) == _refused_text("y2", "Y2('u')")


def test_a_binding_on_the_composition_is_the_compositions():
    composed = (
        fm.residplot(_frame(), x="v", y="w") + fm.Chart(_frame()).mark_point().encode(x="v", y="w")
    ).encode(x2="u")
    assert composed.to_svg()


def test_a_replaced_mark_owns_its_bindings():
    flat = fm.residplot(_frame(), x="v", y="w", **_FLAT)
    assert flat.mark_point().encode(x2="u").to_svg()


@pytest.mark.parametrize("figure", [fm.lmplot, fm.regplot])
@pytest.mark.parametrize("channel", ["x2", "y2"])
def test_lmplot_and_regplot_keep_their_x2(figure, channel):
    assert figure(_frame(), x="v", y="w", **{channel: "u"}).to_svg()


def test_the_refusal_is_a_re_voiceable_entry_refusal():
    expected = _refused_text("x2", "X2('u')")
    with pytest.raises(EntryRefusal) as caught:
        _residplot("layered", x2="u").to_svg()
    err = caught.value
    assert str(err) == expected
    assert str(err.spoken_as("regplot")) == expected.replace("residplot", "regplot")
    # An in-process round trip of a refusal this test raised itself, never
    # data read from outside the test.
    unpickled = pickle.loads(pickle.dumps(err))
    assert str(unpickled) == expected
    assert str(unpickled.spoken_as("regplot")) == expected.replace("residplot", "regplot")
