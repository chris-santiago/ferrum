from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import ferrum as fm


def test_learning_curve_default_emits_direct_labels():
    X, y = load_iris(return_X_y=True)
    chart = fm.learning_curve_chart(
        LogisticRegression(max_iter=200), X, y, cv=3,
    )
    svg = chart.show_svg()
    assert ">Training Score<" in svg
    assert ">Cross-Validation Score<" in svg


def test_learning_curve_legend_suppressed_when_direct_labels():
    X, y = load_iris(return_X_y=True)
    chart = fm.learning_curve_chart(
        LogisticRegression(max_iter=200), X, y, cv=3,
    )
    svg = chart.show_svg()
    # With legend suppression, each split label appears exactly once —
    # the direct-label endpoint annotation. Without suppression we'd get
    # one extra occurrence per series for the legend swatch.
    assert svg.count(">Training Score<") == 1
    assert svg.count(">Cross-Validation Score<") == 1
