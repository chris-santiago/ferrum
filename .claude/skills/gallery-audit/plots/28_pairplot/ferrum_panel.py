"""Ferrum pairplot panel."""
from __future__ import annotations
import os, sys
from pathlib import Path

import polars as pl
from sklearn.datasets import load_iris
import ferrum

W = int(os.environ.get("FERRUM_AUDIT_W", "700"))
H = int(os.environ.get("FERRUM_AUDIT_H", "700"))


def main(out_dir: Path) -> None:
    raw = load_iris(as_frame=True)
    df = pl.from_pandas(raw.frame).with_columns(
        pl.Series("species", [raw.target_names[int(v)] for v in raw.target])
    ).drop("target")
    chart = ferrum.pairplot(df, hue="species")
    (out_dir / "ferrum.svg").write_text(chart.show_svg(), encoding="utf-8")


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); main(out)
