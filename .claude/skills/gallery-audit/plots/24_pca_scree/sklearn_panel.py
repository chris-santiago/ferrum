#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["scikit-learn>=1.4", "matplotlib>=3.7", "numpy>=1.24"]
# ///
"""sklearn PCA scree-plot panel — bar of explained variance ratio + cumulative line.
This is the canonical scree-plot shape; sklearn ships no dedicated `screeplot` so
we render the standard pattern from `PCA.explained_variance_ratio_`."""
from __future__ import annotations
import os, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA

SEED = int(os.environ.get("FERRUM_AUDIT_SEED", "0"))
W = int(os.environ.get("FERRUM_AUDIT_W", "600"))
H = int(os.environ.get("FERRUM_AUDIT_H", "500"))
DPI = int(os.environ.get("FERRUM_AUDIT_DPI", "100"))
plt.rcParams["font.family"] = os.environ.get("FERRUM_AUDIT_FONT", "DejaVu Sans")
plt.rcParams["figure.autolayout"] = False


def main(out_dir: Path) -> None:
    X, _ = load_wine(return_X_y=True)
    pca = PCA(n_components=8, random_state=SEED).fit(X)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    idx = np.arange(1, len(evr) + 1)
    fig, ax = plt.subplots(figsize=(W/DPI, H/DPI), dpi=DPI)
    ax.bar(idx, evr, label="explained variance")
    ax.plot(idx, cum, "-o", color="C1", label="cumulative")
    ax.axhline(0.95, color="grey", linestyle="--", label="95% threshold")
    ax.set_xlabel("Component"); ax.set_ylabel("Explained variance ratio")
    ax.legend()
    fig.savefig(out_dir / "sklearn.png", dpi=DPI); plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); main(out)
