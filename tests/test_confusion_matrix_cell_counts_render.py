import re

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import ferrum as fm


def test_confusion_matrix_default_renders_cell_text():
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200).fit(X, y)
    chart = fm.confusion_matrix_chart(model, X, y)
    svg = chart.show_svg()
    assert re.search(r"<text[^>]*>\s*\d", svg), (
        "no numeric <text> in confusion matrix SVG"
    )
