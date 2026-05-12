#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["yellowbrick>=1.5", "scikit-learn>=1.3,<1.5", "matplotlib>=3.7", "numpy>=1.24"]
# ///
"""yellowbrick KElbow panel (cluster diagnostics)."""
from __future__ import annotations
import os, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer

SEED = int(os.environ.get("FERRUM_AUDIT_SEED", "0"))
W = int(os.environ.get("FERRUM_AUDIT_W", "700"))
H = int(os.environ.get("FERRUM_AUDIT_H", "500"))
DPI = int(os.environ.get("FERRUM_AUDIT_DPI", "100"))
plt.rcParams["font.family"] = os.environ.get("FERRUM_AUDIT_FONT", "DejaVu Sans")
plt.rcParams["figure.autolayout"] = False


def main(out_dir: Path) -> None:
    X, _ = make_blobs(n_samples=300, centers=5, random_state=SEED)
    fig, ax = plt.subplots(figsize=(W/DPI, H/DPI), dpi=DPI)
    viz = KElbowVisualizer(KMeans(n_init=10, random_state=SEED), k=(2, 11), ax=ax)
    viz.fit(X); viz.finalize()
    fig.savefig(out_dir / "yellowbrick.png", dpi=DPI); plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); main(out)
