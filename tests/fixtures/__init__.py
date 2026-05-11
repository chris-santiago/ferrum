"""Phase 10 test fixtures: pre-fit sklearn models serialized via skops.

skops provides type-validated serialization for fitted sklearn estimators;
deserialization refuses any type outside the documented allowlist.

`load_fixture(name)` loads a `.skops` file from `tests/fixtures/models/`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

_FIXTURES_DIR = Path(__file__).parent
_MODELS_DIR = _FIXTURES_DIR / "models"
_DATASETS_DIR = _FIXTURES_DIR / "datasets"


def pinned_sklearn_version() -> str:
    return (_FIXTURES_DIR / "SKLEARN_VERSION").read_text().strip()


# Allowlist for skops.io.load — only sklearn estimators and supporting NumPy types.
ALLOWED_SK_TYPES: tuple[str, ...] = (
    "sklearn.linear_model._logistic.LogisticRegression",
    "sklearn.linear_model._ridge.Ridge",
    "sklearn.linear_model._coordinate_descent.Lasso",
    "sklearn.linear_model._coordinate_descent.ElasticNet",
    "sklearn.ensemble._forest.RandomForestClassifier",
    "sklearn.ensemble._forest.RandomForestRegressor",
    "sklearn.tree._classes.DecisionTreeClassifier",
    "sklearn.tree._classes.DecisionTreeRegressor",
    "sklearn.cluster._kmeans.KMeans",
    "sklearn.decomposition._pca.PCA",
    "sklearn.svm._classes.SVC",
    "sklearn.preprocessing._data.StandardScaler",
    "sklearn.pipeline.Pipeline",
    "numpy.ndarray",
    "numpy.dtype",
    "numpy.dtype[float64]",
    "numpy.dtype[int64]",
    "numpy.dtype[int32]",
)


def load_fixture(name: str) -> Any:
    """Load a fitted model fixture by name (without .skops extension)."""
    import skops.io as sio

    path = _MODELS_DIR / f"{name}.skops"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing fixture {path}. Run `python tests/fixtures/build.py` "
            f"to regenerate (requires scikit-learn=={pinned_sklearn_version()})."
        )
    return sio.load(path, trusted=list(ALLOWED_SK_TYPES))


def load_dataset(name: str):
    """Load a parquet dataset fixture (returns polars.DataFrame)."""
    import polars as pl
    path = _DATASETS_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset {path}. Run `python tests/fixtures/build.py`."
        )
    return pl.read_parquet(path)
