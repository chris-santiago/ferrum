#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["scikit-plot>=0.3","scikit-learn>=1.3,<1.5","scipy>=1.10,<1.12","matplotlib>=3.7","numpy>=1.24,<2"]
# ///
"""scikit-plot lift-curve panel."""
from __future__ import annotations
import os, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

SEED = int(os.environ.get("FERRUM_AUDIT_SEED", "0"))
W = int(os.environ.get("FERRUM_AUDIT_W", "600"))
H = int(os.environ.get("FERRUM_AUDIT_H", "500"))
DPI = int(os.environ.get("FERRUM_AUDIT_DPI", "100"))
plt.rcParams["font.family"] = os.environ.get("FERRUM_AUDIT_FONT", "DejaVu Sans")
plt.rcParams["figure.autolayout"] = False


def main(out_dir: Path) -> None:
    X, y = make_classification(n_samples=500, n_features=10, random_state=SEED)
    model = LogisticRegression(max_iter=2000, random_state=SEED).fit(X, y)
    probas = model.predict_proba(X)
    fig, ax = plt.subplots(figsize=(W/DPI, H/DPI), dpi=DPI)
    skplt.metrics.plot_lift_curve(y, probas, ax=ax)
    fig.savefig(out_dir / "skp.png", dpi=DPI); plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); main(out)
