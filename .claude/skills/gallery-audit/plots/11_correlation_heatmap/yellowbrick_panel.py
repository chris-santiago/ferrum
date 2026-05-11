#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["yellowbrick>=1.5", "scikit-learn>=1.3,<1.5", "matplotlib>=3.7", "numpy>=1.24"]
# ///
"""yellowbrick Rank2D (pearson correlation) panel."""
from __future__ import annotations
import os, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from yellowbrick.features import Rank2D

W = int(os.environ.get("FERRUM_AUDIT_W", "600"))
H = int(os.environ.get("FERRUM_AUDIT_H", "600"))
DPI = int(os.environ.get("FERRUM_AUDIT_DPI", "100"))
plt.rcParams["font.family"] = os.environ.get("FERRUM_AUDIT_FONT", "DejaVu Sans")
plt.rcParams["figure.autolayout"] = False


def main(out_dir: Path) -> None:
    data = load_breast_cancer()  # Bunch with .data ndarray + .feature_names
    X = data.data[:, :10]
    features = list(data.feature_names[:10])
    fig, ax = plt.subplots(figsize=(W/DPI, H/DPI), dpi=DPI)
    viz = Rank2D(features=features, algorithm="pearson", ax=ax)
    viz.fit_transform(X); viz.finalize()
    fig.savefig(out_dir / "yellowbrick.png", dpi=DPI); plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); main(out)
