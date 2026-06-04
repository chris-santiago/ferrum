from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import ferrum as fm


def test_roc_default_emits_auc_text():
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200).fit(X, y)
    chart = fm.roc_chart(model, X, y)
    svg = chart.to_svg()
    assert "AUC = " in svg
