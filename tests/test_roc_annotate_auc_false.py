from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import ferrum as fm


def test_roc_annotate_auc_false_omits_label():
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200).fit(X, y)
    chart = fm.roc_chart(model, X, y, annotate_auc=False)
    svg = chart.show_svg()
    assert "AUC = " not in svg
