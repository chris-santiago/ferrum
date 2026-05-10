"""Phase 9e figure-level function tests."""
import pytest
import polars as pl
import ferrum as fe


def test_all_8_functions_importable():
    assert callable(fe.displot)
    assert callable(fe.catplot)
    assert callable(fe.lmplot)
    assert callable(fe.residplot)
    assert callable(fe.pairplot)
    assert callable(fe.heatmap)
    assert callable(fe.clustermap)
    assert callable(fe.jointplot)


def test_figure_submodule_accessible():
    assert hasattr(fe, "figure")
    assert callable(fe.figure.displot)
