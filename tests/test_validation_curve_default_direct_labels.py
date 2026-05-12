import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import ferrum as fm


def test_validation_curve_default_emits_direct_labels():
    X, y = load_iris(return_X_y=True)
    chart = fm.validation_curve_chart(
        LogisticRegression(max_iter=200),
        X,
        y,
        param="C",
        values=np.logspace(-3, 1, 5),
        cv=3,
    )
    svg = chart.show_svg()
    assert ">Training Score<" in svg
    assert ">Cross-Validation Score<" in svg
    # Legend suppressed — exactly one occurrence per series label
    assert svg.count(">Training Score<") == 1
    assert svg.count(">Cross-Validation Score<") == 1
