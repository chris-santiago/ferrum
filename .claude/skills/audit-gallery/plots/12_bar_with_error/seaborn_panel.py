#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["seaborn>=0.13", "pandas>=2", "matplotlib>=3.7", "numpy>=1.24"]
# ///
"""seaborn barplot panel — defaults include mean + 95% CI."""
from __future__ import annotations
import os, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

W = int(os.environ.get("FERRUM_AUDIT_W", "640"))
H = int(os.environ.get("FERRUM_AUDIT_H", "480"))
DPI = int(os.environ.get("FERRUM_AUDIT_DPI", "100"))
plt.rcParams["font.family"] = os.environ.get("FERRUM_AUDIT_FONT", "DejaVu Sans")
plt.rcParams["figure.autolayout"] = False
FIXTURES = Path(__file__).parent.parent / "_fixtures"


def main(out_dir: Path) -> None:
    df = pd.read_csv(FIXTURES / "tips.csv")
    fig, ax = plt.subplots(figsize=(W/DPI, H/DPI), dpi=DPI)
    sns.barplot(data=df, x="day", y="total_bill", ax=ax)
    fig.savefig(out_dir / "seaborn.png", dpi=DPI); plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); main(out)
