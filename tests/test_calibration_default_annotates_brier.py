from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import ferrum as fm


def test_calibration_default_emits_brier_text():
    X, y = make_classification(n_samples=500, n_classes=2, random_state=0)
    model = LogisticRegression().fit(X, y)
    chart = fm.calibration_chart(model, X=X, y=y)
    svg = chart.to_svg()
    assert "Brier = " in svg


def test_calibration_brier_false_omits_label():
    X, y = make_classification(n_samples=500, n_classes=2, random_state=0)
    model = LogisticRegression().fit(X, y)
    chart = fm.calibration_chart(model, X=X, y=y, annotate_brier=False)
    svg = chart.to_svg()
    assert "Brier = " not in svg
