"""One-shot script to regenerate all Phase 10 model fixtures.

Run with:
    uv run --no-sync python tests/fixtures/build.py

Aborts if installed sklearn doesn't match tests/fixtures/SKLEARN_VERSION.
"""

from __future__ import annotations

import sys
from pathlib import Path

FIXTURES = Path(__file__).parent
MODELS = FIXTURES / "models"
DATASETS = FIXTURES / "datasets"


def _check_sklearn_pin() -> None:
    import sklearn

    pinned = (FIXTURES / "SKLEARN_VERSION").read_text().strip()
    if sklearn.__version__ != pinned:
        print(
            f"ERROR: installed sklearn=={sklearn.__version__} but fixtures "
            f"require sklearn=={pinned}. Run `uv pip install scikit-learn=={pinned}` "
            f"or update tests/fixtures/SKLEARN_VERSION and regenerate all goldens.",
            file=sys.stderr,
        )
        sys.exit(1)


def _save(model, name: str) -> None:
    import skops.io as sio

    path = MODELS / f"{name}.skops"
    sio.dump(model, path)
    print(f"  wrote {path.name}")


def _save_dataset(df, name: str) -> None:
    path = DATASETS / f"{name}.parquet"
    df.write_parquet(path)
    print(f"  wrote {path.name}")


def build_datasets() -> dict:
    import numpy as np
    import polars as pl

    rng = np.random.RandomState(0)

    # Binary classification — 200 rows, 4 features.
    n = 200
    X_bin = rng.randn(n, 4)
    coef = np.array([1.5, -1.0, 0.5, 0.0])
    logits = X_bin @ coef + rng.randn(n) * 0.5
    y_bin = (logits > 0).astype(np.int64)
    bin_df = pl.DataFrame(
        {
            "f0": X_bin[:, 0],
            "f1": X_bin[:, 1],
            "f2": X_bin[:, 2],
            "f3": X_bin[:, 3],
            "y": y_bin,
        }
    )
    _save_dataset(bin_df, "binary_classification")

    # Multiclass classification — 300 rows, 4 features, 3 classes.
    n_mc = 300
    X_mc = rng.randn(n_mc, 4)
    class_means = np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0], [0.0, -1.0, 1.0, 0.0]])
    y_mc = rng.randint(0, 3, size=n_mc)
    X_mc = X_mc + class_means[y_mc]
    mc_df = pl.DataFrame(
        {
            "f0": X_mc[:, 0],
            "f1": X_mc[:, 1],
            "f2": X_mc[:, 2],
            "f3": X_mc[:, 3],
            "y": y_mc.astype(np.int64),
        }
    )
    _save_dataset(mc_df, "multiclass_classification")

    # Regression — 200 rows, 5 features.
    n_reg = 200
    X_reg = rng.randn(n_reg, 5)
    y_reg = X_reg @ np.array([2.0, -1.5, 0.5, 0.0, 0.0]) + rng.randn(n_reg) * 0.3
    reg_df = pl.DataFrame(
        {
            "f0": X_reg[:, 0],
            "f1": X_reg[:, 1],
            "f2": X_reg[:, 2],
            "f3": X_reg[:, 3],
            "f4": X_reg[:, 4],
            "y": y_reg,
        }
    )
    _save_dataset(reg_df, "regression")

    # Clustering — 200 rows, 3 features, 3 well-separated blobs.
    n_clu = 200
    centers = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0]])
    labels = rng.randint(0, 3, size=n_clu)
    X_clu = centers[labels] + rng.randn(n_clu, 3) * 0.5
    clu_df = pl.DataFrame(
        {
            "f0": X_clu[:, 0],
            "f1": X_clu[:, 1],
            "f2": X_clu[:, 2],
        }
    )
    _save_dataset(clu_df, "clustering")

    return {
        "binary": (bin_df, y_bin),
        "multiclass": (mc_df, y_mc),
        "regression": (reg_df, y_reg),
        "clustering": (clu_df, labels),
    }


def build_models(data: dict) -> None:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    bin_df, y_bin = data["binary"]
    X_bin = bin_df.select(["f0", "f1", "f2", "f3"]).to_numpy()

    mc_df, y_mc = data["multiclass"]
    X_mc = mc_df.select(["f0", "f1", "f2", "f3"]).to_numpy()

    reg_df, y_reg = data["regression"]
    X_reg = reg_df.select(["f0", "f1", "f2", "f3", "f4"]).to_numpy()

    clu_df, _ = data["clustering"]
    X_clu = clu_df.to_numpy()

    _save(LogisticRegression(random_state=0, max_iter=500).fit(X_bin, y_bin), "binary_logistic")
    _save(LogisticRegression(random_state=0, max_iter=500).fit(X_mc, y_mc), "multiclass_logistic")
    _save(Ridge(random_state=0).fit(X_reg, y_reg), "regression_ridge")
    _save(RandomForestRegressor(n_estimators=20, random_state=0).fit(X_reg, y_reg), "regression_rf")
    _save(KMeans(n_clusters=3, random_state=0, n_init=10).fit(X_clu), "kmeans_3cluster")
    _save(PCA(n_components=4, random_state=0).fit(X_reg), "pca_4comp")


def main() -> None:
    _check_sklearn_pin()
    MODELS.mkdir(parents=True, exist_ok=True)
    DATASETS.mkdir(parents=True, exist_ok=True)
    print("Building datasets...")
    data = build_datasets()
    print("Building models...")
    build_models(data)
    print("Done.")


if __name__ == "__main__":
    main()
