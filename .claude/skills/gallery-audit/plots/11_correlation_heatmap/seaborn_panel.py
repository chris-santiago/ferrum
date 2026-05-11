#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["seaborn>=0.13", "scikit-learn>=1.4", "pandas>=2", "matplotlib>=3.7", "numpy>=1.24"]
# ///
"""seaborn heatmap panel — defaults only (annot=False is seaborn's default)."""
from __future__ import annotations
import os, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

W = int(os.environ.get("FERRUM_AUDIT_W", "600"))
H = int(os.environ.get("FERRUM_AUDIT_H", "600"))
DPI = int(os.environ.get("FERRUM_AUDIT_DPI", "100"))
plt.rcParams["font.family"] = os.environ.get("FERRUM_AUDIT_FONT", "DejaVu Sans")
plt.rcParams["figure.autolayout"] = False


def main(out_dir: Path) -> None:
    data = load_breast_cancer(as_frame=True)
    corr = data.frame.iloc[:, :10].corr()
    fig, ax = plt.subplots(figsize=(W/DPI, H/DPI), dpi=DPI)
    sns.heatmap(corr, ax=ax)
    fig.savefig(out_dir / "seaborn.png", dpi=DPI); plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); main(out)
