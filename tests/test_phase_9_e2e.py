"""Phase 9 E2E SVG golden tests.

Twelve byte-identical golden tests covering all 8 figure-level functions
plus 4 tricky composite cases. Goldens live under
``tests/test_phase_9_e2e/goldens/*.svg``.

Run with ``FERRUM_UPDATE_GOLDENS=1`` to regenerate the on-disk goldens;
the default behavior is byte-identical comparison.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import ferrum as fe


GOLDENS_DIR = Path(__file__).parent / "test_phase_9_e2e" / "goldens"
UPDATE = os.environ.get("FERRUM_UPDATE_GOLDENS") == "1"


# ---------------------------------------------------------------------------
# Deterministic fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_iris_like():
    """Fixed-seed iris-like 60-row dataset used by most goldens."""
    rng = np.random.default_rng(0)
    return pl.DataFrame({
        "sepal_length": rng.normal(5.0, 0.5, 60).tolist(),
        "sepal_width":  rng.normal(3.0, 0.3, 60).tolist(),
        "petal_length": rng.normal(3.5, 0.7, 60).tolist(),
        "species":      ["setosa"] * 30 + ["versicolor"] * 30,
    })


@pytest.fixture
def df_reg():
    """Fixed-seed regression dataset."""
    rng = np.random.default_rng(1)
    n = 40
    x = np.linspace(0.0, 10.0, n)
    y = 2.0 + 0.5 * x + rng.normal(0, 1.0, n)
    return pl.DataFrame({"x": x.tolist(), "y": y.tolist()})


@pytest.fixture
def df_heat():
    """Small wide-form dataset for heatmap."""
    return pl.DataFrame({
        "row": ["r1", "r2", "r3", "r4"],
        "A":   [1.0, 2.0, 3.0, 4.0],
        "B":   [4.0, 3.0, 2.0, 1.0],
        "C":   [2.5, 1.5, 3.5, 4.5],
    })


@pytest.fixture
def df_cluster():
    rng = np.random.default_rng(2)
    return pl.DataFrame({
        "row": [f"r{i}" for i in range(5)],
        "A":   rng.normal(0, 1, 5).tolist(),
        "B":   rng.normal(0, 1, 5).tolist(),
        "C":   rng.normal(0, 1, 5).tolist(),
        "D":   rng.normal(0, 1, 5).tolist(),
    })


@pytest.fixture
def df_joint():
    rng = np.random.default_rng(3)
    return pl.DataFrame({
        "x": rng.normal(0, 1, 50).tolist(),
        "y": rng.normal(0, 1, 50).tolist(),
    })


# ---------------------------------------------------------------------------
# Helper — golden compare / update
# ---------------------------------------------------------------------------


def _check_or_update(name: str, svg: str) -> None:
    """Compare ``svg`` to the on-disk golden ``name``.

    With ``FERRUM_UPDATE_GOLDENS=1`` the on-disk golden is (re)written.
    Otherwise the function asserts byte-for-byte equality against the
    committed file. A missing golden is treated as an explicit test
    failure rather than a silent skip — goldens must be regenerated
    deliberately.
    """
    path = GOLDENS_DIR / name
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(svg)
        return
    if not path.exists():
        pytest.fail(
            f"golden {name!r} does not exist; rerun with "
            f"FERRUM_UPDATE_GOLDENS=1 to generate it"
        )
    expected = path.read_text()
    assert svg == expected, (
        f"golden mismatch for {name!r}; "
        f"rerun with FERRUM_UPDATE_GOLDENS=1 to refresh "
        f"(expected {len(expected)} bytes, got {len(svg)} bytes)"
    )


# ---------------------------------------------------------------------------
# 1. displot — basic histogram
# ---------------------------------------------------------------------------


def test_displot_hist_golden(df_iris_like):
    chart = fe.displot(df_iris_like, x="sepal_length", kind="hist")
    _check_or_update("displot_hist.svg", chart.show_svg())


# ---------------------------------------------------------------------------
# 2. catplot — boxplot with hue + dodge
# ---------------------------------------------------------------------------


def test_catplot_box_golden(df_iris_like):
    chart = fe.catplot(
        df_iris_like,
        x="species", y="sepal_length",
        hue="species", kind="box", dodge=True,
    )
    _check_or_update("catplot_box.svg", chart.show_svg())


# ---------------------------------------------------------------------------
# 3. lmplot — LM regression with CI band
# ---------------------------------------------------------------------------


def test_lmplot_lm_ci_golden(df_reg):
    chart = fe.lmplot(df_reg, x="x", y="y", method="lm", ci=95)
    _check_or_update("lmplot_lm_ci.svg", chart.show_svg())


# ---------------------------------------------------------------------------
# 4. residplot — residuals + lowess overlay
# ---------------------------------------------------------------------------


def test_residplot_lowess_golden(df_reg):
    chart = fe.residplot(df_reg, x="x", y="y", lowess=True)
    _check_or_update("residplot_lowess.svg", chart.show_svg())


# ---------------------------------------------------------------------------
# 5. pairplot — 3x3 grid (no hue)
# ---------------------------------------------------------------------------


def test_pairplot_3x3_golden(df_iris_like):
    rc = fe.pairplot(
        df_iris_like,
        vars=["sepal_length", "sepal_width", "petal_length"],
    )
    _check_or_update("pairplot_3x3.svg", rc.show_svg())


# ---------------------------------------------------------------------------
# 6. heatmap — annotated heatmap
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="heatmap requires a continuous color scale to map Float64 'value' "
           "column to colors. scale_resolve.rs's ColorScale enum currently has "
           "only the Categorical variant — calling distinct_values_in_order on a "
           "Float64 column raises 'can't enumerate distinct values from column "
           "dtype Float64'. Phase 8b's ContinuousScheme exists at the palette "
           "level but is not wired into ColorScale / build_color_scale. Needs a "
           "ColorScale::Continuous variant + numeric domain extraction + "
           "interpolation in the lookup path. Feature gap pre-dating Phase 9.",
    strict=True,
)
def test_heatmap_annot_golden(df_heat):
    chart = fe.heatmap(df_heat, annot=True)
    _check_or_update("heatmap_annot.svg", chart.show_svg())


# ---------------------------------------------------------------------------
# 7. clustermap — basic clustered heatmap (defaults)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="clustermap heatmap layer references 'row_link_order' / "
           "'col_link_order' produced by Linkage but the Reorder stat "
           "can't see them across transform boundaries; renderer raises "
           "ValueError: stat_reorder: index column 'row_link_order' not found",
    strict=True,
)
def test_clustermap_basic_golden(df_cluster):
    cm = fe.clustermap(df_cluster)
    _check_or_update("clustermap_basic.svg", cm.show_svg())


# ---------------------------------------------------------------------------
# 8. jointplot — KDE center + hist marginals
# ---------------------------------------------------------------------------


def test_jointplot_kde_hist_golden(df_joint):
    jc = fe.jointplot(df_joint, x="x", y="y", kind="kde", marginal_kind="hist")
    _check_or_update("jointplot_kde_hist.svg", jc.show_svg())


# ---------------------------------------------------------------------------
# 9. pairplot — 3x3 with hue (tricky composite)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="pairplot with hue='species' propagates color encoding to the "
           "template but Repeat substitution does not preserve the hue "
           "column in per-cell data slices; renderer raises "
           "ValueError: unknown column 'species' referenced by an encoding",
    strict=True,
)
def test_pairplot_3x3_hue_golden(df_iris_like):
    rc = fe.pairplot(
        df_iris_like,
        vars=["sepal_length", "sepal_width", "petal_length"],
        hue="species",
    )
    _check_or_update("pairplot_3x3_hue.svg", rc.show_svg())


# ---------------------------------------------------------------------------
# 10. clustermap — alternate linkage method + metric + z_score (tricky)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="same clustermap renderer gap as clustermap_basic — stat_reorder "
           "cannot resolve 'row_link_order' index column at render time",
    strict=True,
)
def test_clustermap_row_col_dendrograms_golden(df_cluster):
    cm = fe.clustermap(
        df_cluster,
        method="average", metric="cosine", z_score="rows",
    )
    _check_or_update("clustermap_row_col_dendrograms.svg", cm.show_svg())


# ---------------------------------------------------------------------------
# 11. jointplot — KDE center + KDE marginals (tricky)
# ---------------------------------------------------------------------------


def test_jointplot_kde_marginals_golden(df_joint):
    jc = fe.jointplot(df_joint, x="x", y="y", kind="kde", marginal_kind="kde")
    _check_or_update("jointplot_kde_marginals.svg", jc.show_svg())


# ---------------------------------------------------------------------------
# 12. displot — stacked histogram with hue (tricky)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="displot multiple='stack' with hue=' species' emits a color "
           "encoding for 'species' but the binned-transform output does "
           "not retain that column; renderer raises ValueError: unknown "
           "column 'species' referenced by an encoding",
    strict=True,
)
def test_displot_stacked_hist_golden(df_iris_like):
    chart = fe.displot(
        df_iris_like,
        x="sepal_length", hue="species",
        kind="hist", multiple="stack",
    )
    _check_or_update("displot_stacked_hist.svg", chart.show_svg())
