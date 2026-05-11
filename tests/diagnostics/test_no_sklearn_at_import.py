"""Regression test for Phase 10 done-criterion:
sklearn must NOT be imported by `import ferrum` or `ModelSource.__init__`.
"""
from __future__ import annotations

import sys


class _DuckModel:
    """Non-sklearn duck-typed model with predict()."""
    def predict(self, X):
        return [0] * len(X)


def _drop_sklearn():
    for mod in list(sys.modules):
        if mod == "sklearn" or mod.startswith("sklearn."):
            del sys.modules[mod]


def test_import_ferrum_does_not_load_sklearn():
    _drop_sklearn()
    assert "sklearn" not in sys.modules

    import ferrum  # noqa: F401
    assert "sklearn" not in sys.modules, (
        "sklearn loaded as a side-effect of `import ferrum`"
    )


def test_modelsource_init_does_not_load_sklearn():
    _drop_sklearn()

    import polars as pl
    import ferrum

    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    source = ferrum.ModelSource(_DuckModel(), df)
    assert "sklearn" not in sys.modules, (
        "sklearn loaded as a side-effect of ModelSource(...)"
    )
    assert source is not None


def test_require_sklearn_raises_clear_message_when_missing(monkeypatch):
    """Simulates sklearn missing — ImportError must mention `ferrum[models]`."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ImportError("No module named 'sklearn'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from ferrum._diagnostics.deps import require_sklearn
    import pytest
    with pytest.raises(ImportError, match=r"ferrum\[models\]|pip install scikit-learn"):
        require_sklearn("predictions")
