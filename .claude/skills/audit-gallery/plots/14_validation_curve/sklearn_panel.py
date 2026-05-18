#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["scikit-learn>=1.4", "matplotlib>=3.7", "numpy>=1.24"]
# ///
"""sklearn validation-curve panel."""
from __future__ import annotations
import os, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import ValidationCurveDisplay

SEED = int(os.environ.get("FERRUM_AUDIT_SEED", "0"))
W = int(os.environ.get("FERRUM_AUDIT_W", "720"))
H = int(os.environ.get("FERRUM_AUDIT_H", "480"))
DPI = int(os.environ.get("FERRUM_AUDIT_DPI", "100"))
plt.rcParams["font.family"] = os.environ.get("FERRUM_AUDIT_FONT", "DejaVu Sans")
plt.rcParams["figure.autolayout"] = False
ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def main(out_dir: Path) -> None:
    X, y = load_diabetes(return_X_y=True)
    model = Ridge(random_state=SEED)
    fig, ax = plt.subplots(figsize=(W/DPI, H/DPI), dpi=DPI)
    ValidationCurveDisplay.from_estimator(
        model, X, y, param_name="alpha", param_range=ALPHAS, ax=ax,
    )
    fig.savefig(out_dir / "sklearn.png", dpi=DPI); plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); main(out)
