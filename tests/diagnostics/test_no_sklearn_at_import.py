"""sklearn is optional: ``import ferrum`` and every sklearn-free path work without it.

The check runs in a fresh interpreter (``_SKLEARN_FREE_PATHS``) whose first act
is to install an import blocker that makes ``import sklearn`` and every
``sklearn.*`` import raise ``ImportError``, as if sklearn were not installed.
It then imports ferrum and runs every path that must not need sklearn. The
blocker also records each sklearn import it refuses, so an import swallowed
by a ``try``/``except ImportError`` still fails the check. The pytest process
itself never loses or reloads a module.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

_SKLEARN_FREE_PATHS = textwrap.dedent(
    """
    import importlib.abc
    import json
    import sys

    refused = []


    class _NoSklearn(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "sklearn" or name.startswith("sklearn."):
                refused.append(name)
                raise ImportError(f"No module named {name!r} (blocked by the test)")
            return None


    assert not any(m == "sklearn" or m.startswith("sklearn.") for m in sys.modules)
    sys.meta_path.insert(0, _NoSklearn())

    import numpy as np
    import polars as pl

    import ferrum
    from ferrum.diagnostics._internal.precomputed import _PrecomputedSource

    ran = ["import ferrum"]


    class _DuckModel:
        def predict(self, X):
            return [0] * len(X)


    assert ferrum.ModelSource(_DuckModel(), pl.DataFrame({"a": [1.0, 2.0, 3.0]})) is not None
    ran.append("ModelSource(duck model)")

    chart = ferrum.classification_report_chart(y_true=[0, 0, 1, 1], y_pred=[0, 1, 1, 1])
    assert chart.to_svg()
    ran.append("classification_report_chart(y_true=, y_pred=)")

    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=60)
    y_score = np.clip(rng.standard_normal(60) * 0.4 + y_true * 0.8, 0.0, 1.0)
    y_hard = (y_score >= 0.5).astype(int)
    binary_soft = _PrecomputedSource(y_true, y_score)
    binary_hard = _PrecomputedSource(y_true, y_hard)

    rng = np.random.default_rng(99)
    m_true = rng.integers(0, 3, size=75)
    logits = rng.standard_normal((75, 3))
    m_prob = np.exp(logits - logits.max(axis=1, keepdims=True))
    m_prob /= m_prob.sum(axis=1, keepdims=True)
    multi_soft = _PrecomputedSource(m_true, m_prob)
    multi_hard = _PrecomputedSource(m_true, m_prob.argmax(axis=1))
    classes = {str(c) for c in np.unique(m_true).tolist()}

    df = binary_soft.roc_curve()
    assert {"fpr", "tpr", "threshold"} <= set(df.columns) and len(df) > 2
    assert set(multi_soft.roc_curve()["class"].unique().to_list()) == classes
    ran.append("roc_curve")

    df = binary_soft.pr_curve()
    assert {"precision", "recall"} <= set(df.columns) and len(df) > 2
    assert set(multi_soft.pr_curve()["class"].unique().to_list()) == classes
    ran.append("pr_curve")

    df = binary_soft.calibration_curve()
    assert {"mean_predicted", "fraction_positive"} <= set(df.columns) and len(df) >= 1
    assert 1 <= len(binary_soft.calibration_curve(n_bins=5, strategy="quantile")) <= 5
    ran.append("calibration_curve")

    df = binary_hard.confusion_matrix()
    assert {"actual", "predicted", "value", "value_fmt"} <= set(df.columns) and len(df) == 4
    assert len(multi_hard.confusion_matrix()) == 9
    ran.append("confusion_matrix")

    df = binary_soft.discrimination_threshold(n_thresholds=20)
    assert {"threshold", "precision", "recall", "f1", "queue_rate"} <= set(df.columns)
    assert len(df) == 20
    assert len(binary_soft.discrimination_threshold(cv=None, n_thresholds=10)) == 10
    try:
        binary_soft.discrimination_threshold(cv=3)
    except ValueError as exc:
        assert "cv=" in str(exc)
    else:
        raise AssertionError("discrimination_threshold(cv=3) did not refuse")
    ran.append("discrimination_threshold")

    print(json.dumps({"ran": ran, "refused": refused}))
    """
)

_EXPECTED_PATHS = [
    "import ferrum",
    "ModelSource(duck model)",
    "classification_report_chart(y_true=, y_pred=)",
    "roc_curve",
    "pr_curve",
    "calibration_curve",
    "confusion_matrix",
    "discrimination_threshold",
]


def test_ferrum_and_its_sklearn_free_paths_run_with_sklearn_uninstallable() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _SKLEARN_FREE_PATHS],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"an sklearn-free path failed with sklearn blocked:\n{result.stderr}"
    )
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["ran"] == _EXPECTED_PATHS
    assert report["refused"] == [], (
        f"an sklearn-free path tried to import sklearn: {report['refused']}"
    )


def test_require_sklearn_raises_clear_message_when_missing(monkeypatch):
    """Simulates sklearn missing — ImportError must mention `ferrum[models]`."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ImportError("No module named 'sklearn'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from ferrum.diagnostics._internal.deps import require_sklearn

    with pytest.raises(ImportError, match=r"ferrum\[models\]|pip install scikit-learn"):
        require_sklearn("predictions")
