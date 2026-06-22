"""Phase 8b Task 37: continuous_palette() lookup + Gradient factory."""

from __future__ import annotations

import pytest

import ferrum as fe


def test_viridis_lookup():
    s = fe.continuous_palette("viridis")
    assert s is not None


def test_plasma_lookup():
    fe.continuous_palette("plasma")


def test_magma_lookup():
    fe.continuous_palette("magma")


def test_inferno_lookup():
    fe.continuous_palette("inferno")


def test_cividis_lookup():
    fe.continuous_palette("cividis")


def test_unknown_palette_raises():
    with pytest.raises(ValueError, match="Unknown colormap"):
        fe.continuous_palette("notacolor")


def test_continuous_palette_list():
    # The list is derived from the Rust palette registry (all sequential +
    # diverging schemes), so it covers the classic colormaps and the custom
    # continuous palettes without drifting from what from_name() accepts.
    from ferrum._core import list_palettes, palette_kind

    names = set(fe.continuous_palette.list())
    expected = {n for n in list_palettes() if palette_kind(n) in ("sequential", "diverging")}
    assert names == expected
    # The classic colormaps are always present.
    assert {"viridis", "plasma", "magma", "inferno", "cividis"} <= names
    # Every listed name must be constructible.
    for n in names:
        assert fe.continuous_palette(n) is not None


def test_reversed_returns_new_scheme():
    s = fe.continuous_palette("viridis")
    rev = s.reversed()
    assert rev is not s


def test_gradient_two_stops():
    g = fe.Gradient([(0.0, "red"), (1.0, "blue")])
    assert g is not None
