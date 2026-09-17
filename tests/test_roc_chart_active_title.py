"""ROC chart titles are plain names, not metric carriers (#219 / D4).

Before 2026-09-13 a single-curve ``roc_chart`` titled itself
``"ROC Curve — AUC 0.984"`` while the endpoint overlay printed the same
number a second time, and the title did so whether or not ``annotate_auc``
was set. The title now keeps its name in every shape, and the overlay is
the only metric surface.
"""

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import ferrum as fm


def test_roc_single_curve_title_is_the_plain_name():
    X, y = make_classification(n_samples=200, n_classes=2, random_state=0)
    model = LogisticRegression().fit(X, y)
    chart = fm.roc_chart(model, X, y, per_class=False)
    assert chart._title.text == "ROC Curve"
    svg = chart.to_svg()
    assert ">ROC Curve<" in svg
    assert "ROC Curve — AUC" not in svg


def test_roc_per_class_title_is_the_same_plain_name():
    X, y = make_classification(
        n_samples=200,
        n_classes=3,
        n_informative=3,
        random_state=0,
    )
    model = LogisticRegression(max_iter=300).fit(X, y)
    chart = fm.roc_chart(model, X, y, per_class=True)
    assert chart._title.text == "ROC Curve"
    assert ">ROC Curve<" in chart.to_svg()
