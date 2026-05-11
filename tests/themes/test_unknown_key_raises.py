"""Theme unknown-key validation at construction."""
import pytest

import ferrum as fm


def test_unknown_key_raises_at_construction() -> None:
    with pytest.raises(ValueError) as excinfo:
        fm.Theme(typo_key="foo")
    msg = str(excinfo.value)
    assert "Unknown Theme key" in msg
    assert "typo_key" in msg


def test_multiple_unknown_keys_listed() -> None:
    with pytest.raises(ValueError) as excinfo:
        fm.Theme(typo_a="x", typo_b="y", font_family="DejaVu Sans")
    msg = str(excinfo.value)
    assert "typo_a" in msg
    assert "typo_b" in msg
    # Known key not mentioned in error.
    assert "font_family" not in msg


def test_known_keys_accepted() -> None:
    # Sample drawn across the spec — proves the set covers the breadth.
    t = fm.Theme(
        background="#ffffff",
        font_family="DejaVu Sans",
        title_anchor="start",
        grid=True,
        grid_dash=[3, 3],
        color_scheme="tableau10",
        legend_orient="bottom",
    )
    assert t is not None
