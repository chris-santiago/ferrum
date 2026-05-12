"""Ferrum parallel-coordinates panel."""
from __future__ import annotations
import os, sys
from pathlib import Path

import polars as pl
from sklearn.datasets import load_iris
import ferrum

SEED = int(os.environ.get("FERRUM_AUDIT_SEED", "0"))
W = int(os.environ.get("FERRUM_AUDIT_W", "800"))
H = int(os.environ.get("FERRUM_AUDIT_H", "500"))


def main(out_dir: Path) -> None:
    raw = load_iris(as_frame=True)
    df = pl.from_pandas(raw.frame).with_columns(
        pl.Series("species", [raw.target_names[int(v)] for v in raw.target])
    ).drop("target")
    chart = ferrum.parallel_coordinates_chart(df, hue="species").properties(width=W, height=H)
    (out_dir / "ferrum.svg").write_text(chart.show_svg(), encoding="utf-8")


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); main(out)
