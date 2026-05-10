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
    names = fe.continuous_palette.list()
    assert set(names) == {"viridis", "plasma", "magma", "inferno", "cividis"}


def test_reversed_returns_new_scheme():
    s = fe.continuous_palette("viridis")
    rev = s.reversed()
    assert rev is not s


def test_gradient_two_stops():
    g = fe.Gradient([(0.0, "red"), (1.0, "blue")])
    assert g is not None
