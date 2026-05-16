"""Theme fallback chain resolution in to_spec_dict()."""

import ferrum as fm


def test_title_color_falls_back_to_font_color() -> None:
    t = fm.Theme(font_color="#222222")
    d = t.to_spec_dict()
    assert d["font_color"] == "#222222"
    assert d["title_color"] == "#222222"
    assert d["label_color"] == "#222222"


def test_explicit_title_color_overrides_fallback() -> None:
    t = fm.Theme(font_color="#222222", title_color="#ff0000")
    d = t.to_spec_dict()
    assert d["title_color"] == "#ff0000"
    assert d["label_color"] == "#222222"


def test_font_family_chain() -> None:
    t = fm.Theme(font_family="DejaVu Sans")
    d = t.to_spec_dict()
    assert d["title_font_family"] == "DejaVu Sans"
    assert d["label_font_family"] == "DejaVu Sans"


def test_no_fallback_when_source_also_unset() -> None:
    t = fm.Theme(mark_color="#000000")
    d = t.to_spec_dict()
    assert "title_color" not in d
    assert "label_color" not in d
    assert "title_font_family" not in d
    assert "label_font_family" not in d
