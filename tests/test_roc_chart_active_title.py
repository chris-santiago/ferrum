import re

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import ferrum as fm


def test_roc_single_curve_active_title():
    X, y = make_classification(n_samples=200, n_classes=2, random_state=0)
    model = LogisticRegression().fit(X, y)
    chart = fm.roc_chart(model, X, y, per_class=False)
    svg = chart.show_svg()
    assert re.search(r">ROC Curve — AUC \d\.\d{3}<", svg)


def test_roc_per_class_falls_back_to_descriptive_title():
    X, y = make_classification(
        n_samples=200, n_classes=3, n_informative=3, random_state=0,
    )
    model = LogisticRegression(max_iter=300).fit(X, y)
    chart = fm.roc_chart(model, X, y, per_class=True)
    svg = chart.show_svg()
    assert ">ROC Curve<" in svg
