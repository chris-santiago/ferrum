"""Every channel's `title=` and `legend=` either works or is refused, never warned and dropped.

Whether a channel draws a legend is one fact, `ChannelBase._legend_surface`
(`own`, `cascade` for X/Y, `none`). The honored-kwarg bits for `legend` and for
the legend's own `title` derive from it, and an un-honored `title=` refuses
(REFUSE MALFORMED). This matrix runs every `ChannelBase` subclass through
`title=`, a legend dict, `legend=None` and `legend=False`:

- a cell that works puts the value on the wire (`to_encoding_spec_dict`);
- a cell that is refused raises a typed `ValueError` naming the channel and the
  value;
- no cell warns.

`legend=None`/`legend=False` on a channel with no legend is the silent "no legend
here" idiom: it is accepted and puts nothing on the wire.
"""

from __future__ import annotations

import warnings

import pytest

import ferrum as fm
from ferrum._core import aux_legend_channels
from ferrum.encoding.base import ChannelBase


def _all_subclasses(cls: type) -> list[type]:
    out = []
    for sub in cls.__subclasses__():
        out.append(sub)
        out.extend(_all_subclasses(sub))
    return out


CHANNELS = sorted(
    {c for c in _all_subclasses(ChannelBase) if getattr(fm, c.__name__, None) is c},
    key=lambda c: c.__name__,
)
CELLS = {
    "title": ({"title": "T"}, "title"),
    "legend_dict": ({"legend": {"title": "T"}}, "legend"),
    "legend_none": ({"legend": None}, "legend"),
    "legend_false": ({"legend": False}, "legend"),
}


def test_population_covers_every_public_channel():
    assert len(CHANNELS) == 34
    assert {"Color", "X", "Opacity", "StrokeWidth", "Detail", "FacetRow"} <= {
        c.__name__ for c in CHANNELS
    }


def _construct(cls: type, kwargs: dict):
    from ferrum._warn import reset_warnings

    reset_warnings()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return cls("g", **kwargs)


@pytest.mark.parametrize("cell", sorted(CELLS))
@pytest.mark.parametrize("cls", CHANNELS, ids=lambda c: c.__name__)
def test_works_or_refuses(cls, cell):
    kwargs, key = CELLS[cell]
    silent_no_legend = cell in ("legend_none", "legend_false") and cls._legend_surface == "none"
    if key in cls._honored_kwargs or silent_no_legend:
        wire = _construct(cls, kwargs).to_encoding_spec_dict()
        if cell == "title":
            assert wire["title"] == "T"
        elif cell == "legend_dict":
            assert wire["legend"] == {"title": "T"}
        elif cls._legend_surface == "own":
            # A legend owner's bare None/False suppresses its own legend.
            assert wire["legend"] == {"disabled": True}
        else:
            # `cascade` (X/Y) and `none`: nothing of its own to suppress.
            assert "legend" not in wire
    else:
        with pytest.raises(ValueError, match=rf"^{cls.__name__}\({key}=.*\) is not "):
            _construct(cls, kwargs)


@pytest.mark.parametrize(
    "cls", [c for c in CHANNELS if c._legend_surface == "none"], ids=lambda c: c.__name__
)
@pytest.mark.parametrize("value", [None, False])
def test_no_legend_channel_puts_nothing_on_the_wire(cls, value):
    assert _construct(cls, {"legend": value}).to_encoding_spec_dict() == {"field": "g"}


def test_vacuity_control_a_legend_owner_emits_its_legend_and_title():
    wire = _construct(fm.Color, {"title": "T", "legend": None}).to_encoding_spec_dict()
    assert wire["title"] == "T"
    assert wire["legend"] == {"disabled": True}


def test_title_refusal_text_names_channel_and_value():
    with pytest.raises(ValueError) as exc:
        _construct(fm.Angle, {"title": "T"})
    assert str(exc.value) == (
        "Angle(title='T') is not honored: Angle draws no legend, and "
        "title= is not supported on this channel."
    )


def test_legend_surfaces_agree_with_the_published_aux_legend_list():
    own = {c._channel_name for c in CHANNELS if c._legend_surface == "own"}
    assert set(aux_legend_channels()) <= own
    assert own == {
        "color",
        "fill",
        "stroke",
        "size",
        "shape",
        "stroke_dash",
        "opacity",
        "fill_opacity",
        "stroke_opacity",
        "stroke_width",
    }


@pytest.mark.parametrize("cls", CHANNELS, ids=lambda c: c.__name__)
def test_honored_legend_and_title_follow_the_surface(cls):
    surface = cls._legend_surface
    assert ("legend" in cls._honored_kwargs) == (surface != "none")
    if surface == "own":
        assert "title" in cls._honored_kwargs
