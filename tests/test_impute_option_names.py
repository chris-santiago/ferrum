"""The impute option names are one list, read by the accepted set, the refusal and the registry."""

from __future__ import annotations

import pytest

import ferrum as fm
from ferrum._core import transform_registry
from ferrum._validate import IMPUTE_OPTION_NAMES

_NOT_OPTIONS = {"field", "groupby", "key", "name"}


def test_the_engines_registry_publishes_exactly_the_option_names_in_their_order():
    published = [p for p in transform_registry("impute")["params"] if p not in _NOT_OPTIONS]
    assert published == list(IMPUTE_OPTION_NAMES)


@pytest.mark.parametrize("door", ["Y", "X"])
def test_an_unknown_option_is_refused_naming_every_option_in_order(door: str):
    with pytest.raises(ValueError) as excinfo:
        getattr(fm, door)("v", impute={"bogus": 1})
    assert "unknown option 'bogus'; expected method|value|keyvals|frame" in str(excinfo.value)
