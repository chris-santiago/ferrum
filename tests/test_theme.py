import pytest

from ferrum.themes import Theme


def test_theme_default_has_no_props():
    t = Theme()
    assert t._props == {}


def test_theme_with_kwargs_stores_them():
    t = Theme(background="#000", font_family="Inter")
    assert t._props == {"background": "#000", "font_family": "Inter"}


def test_theme_omits_none_values():
    t = Theme(background="#000", font_family=None)
    assert t._props == {"background": "#000"}


def test_theme_update_returns_new_theme_with_merged_props():
    t1 = Theme(background="#000")
    t2 = t1.update(font_family="Inter")
    assert t1._props == {"background": "#000"}
    assert t2._props == {"background": "#000", "font_family": "Inter"}
    assert t1 is not t2


def test_theme_update_overrides_existing_prop():
    t1 = Theme(background="#000")
    t2 = t1.update(background="#fff")
    assert t1._props == {"background": "#000"}
    assert t2._props == {"background": "#fff"}


def test_theme_eq_when_props_match():
    t1 = Theme(background="#000", font_family="Inter")
    t2 = Theme(font_family="Inter", background="#000")
    assert t1 == t2


def test_theme_to_theme_inputs_dict_passes_through_props():
    t = Theme(background="#1a1a2e", font_color="#e6e6e6")
    d = t.to_theme_inputs_dict()
    assert d["background"] == "#1a1a2e"
    assert d["font_color"] == "#e6e6e6"


def test_theme_hashable():
    t = Theme(background="#000")
    s = {t}
    assert t in s


def test_eight_builtins_exist():
    from ferrum.themes import (default, minimal, dark, publication,
                                 economist, fivethirtyeight,
                                 solarized_light, solarized_dark)
    for t in (default, minimal, dark, publication, economist,
              fivethirtyeight, solarized_light, solarized_dark):
        assert isinstance(t, Theme)


def test_default_theme_has_no_props():
    from ferrum.themes import default
    assert default._props == {}


def test_dark_theme_has_dark_background():
    from ferrum.themes import dark
    assert dark._props["background"] == "#1a1a2e"
