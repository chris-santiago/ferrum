#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "yellowbrick>=1.5",
#   "scikit-learn>=1.3,<1.5",
#   "matplotlib>=3.7",
#   "numpy>=1.24",
# ]
# ///
"""yellowbrick ROC panel — isolated PEP 723 env. Pinned sklearn<1.5 because
yellowbrick has known incompatibilities with newer sklearn private APIs."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ROCAUC

SEED = int(os.environ.get("FERRUM_AUDIT_SEED", "0"))
W = int(os.environ.get("FERRUM_AUDIT_W", "640"))
H = int(os.environ.get("FERRUM_AUDIT_H", "480"))
DPI = int(os.environ.get("FERRUM_AUDIT_DPI", "100"))
FONT = os.environ.get("FERRUM_AUDIT_FONT", "DejaVu Sans")

plt.rcParams["font.family"] = FONT
plt.rcParams["figure.autolayout"] = False


def main(out_dir: Path) -> None:
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED)
    model = LogisticRegression(max_iter=2000, random_state=SEED)

    fig, ax = plt.subplots(figsize=(W / DPI, H / DPI), dpi=DPI)
    viz = ROCAUC(model, ax=ax)
    viz.fit(Xtr, ytr)
    viz.score(Xte, yte)
    viz.finalize()  # use defaults; do NOT call viz.show() (it tight_layouts)
    fig.savefig(out_dir / "yellowbrick.png", dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    main(out)
