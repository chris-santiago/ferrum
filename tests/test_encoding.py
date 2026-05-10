import warnings
import pytest

from ferrum._warn import reset_warnings
from ferrum.encoding.base import ChannelBase


class _TestChannel(ChannelBase):
    _channel_name = "x"
    _renders_in_phase_8a = True
    _honored_kwargs = frozenset(["type", "scale", "title"])


def test_channelbase_stores_field_and_kwargs():
    reset_warnings()
    c = _TestChannel("price", type="Q", title="Price")
    assert c.field == "price"
    assert c._kwargs == {"type": "Q", "title": "Price"}


def test_channelbase_warns_once_on_deferred_kwarg():
    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _TestChannel("price", axis={"grid": False})
    assert len(w) == 1
    assert "axis" in str(w[0].message)


def test_to_encoding_spec_dict_has_field_and_type():
    reset_warnings()
    c = _TestChannel("price", type="Q")
    d = c.to_encoding_spec_dict()
    assert d["field"] == "price"
    assert d["type_"] == "Q"


def test_to_implicit_transforms_with_bin_kwarg():
    class _BinTestChannel(ChannelBase):
        _channel_name = "x"
        _renders_in_phase_8a = True
        _honored_kwargs = frozenset(["type", "bin", "aggregate"])

    reset_warnings()
    c = _BinTestChannel("price", bin=True)
    transforms = c.to_implicit_transforms()
    assert len(transforms) == 1
    # First (and only) transform should be a Bin instance
    from ferrum import Bin
    assert isinstance(transforms[0], Bin)


def test_to_implicit_transforms_with_aggregate_kwarg():
    class _AggTestChannel(ChannelBase):
        _channel_name = "y"
        _renders_in_phase_8a = True
        _honored_kwargs = frozenset(["type", "aggregate"])

    reset_warnings()
    c = _AggTestChannel("latency", aggregate="mean")
    transforms = c.to_implicit_transforms()
    assert len(transforms) == 1
    from ferrum import Aggregate
    assert isinstance(transforms[0], Aggregate)
