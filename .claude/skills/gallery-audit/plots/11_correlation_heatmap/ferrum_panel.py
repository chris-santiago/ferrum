"""Ferrum correlation heatmap panel."""
from __future__ import annotations
import os, sys
from pathlib import Path

import polars as pl
from sklearn.datasets import load_breast_cancer
import ferrum

W = int(os.environ.get("FERRUM_AUDIT_W", "600"))
H = int(os.environ.get("FERRUM_AUDIT_H", "600"))


def main(out_dir: Path) -> None:
    data = load_breast_cancer(as_frame=True)
    # First 10 numeric features → 10×10 correlation matrix.
    df = data.frame.iloc[:, :10]
    corr = df.corr().reset_index().rename(columns={"index": "feature"})
    # ferrum.heatmap reads a wide DataFrame where the first non-numeric column
    # becomes row labels; remaining numeric columns become heatmap columns.
    pcorr = pl.from_pandas(corr)
    chart = ferrum.heatmap(pcorr).properties(width=W, height=H)
    (out_dir / "ferrum.svg").write_text(chart.show_svg(), encoding="utf-8")


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); main(out)
