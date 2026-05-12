"""Ferrum jointplot panel."""
from __future__ import annotations
import os, sys
from pathlib import Path

import polars as pl
import ferrum

W = int(os.environ.get("FERRUM_AUDIT_W", "600"))
H = int(os.environ.get("FERRUM_AUDIT_H", "600"))
FIXTURES = Path(__file__).parent.parent / "_fixtures"


def main(out_dir: Path) -> None:
    df = pl.read_csv(FIXTURES / "tips.csv")
    chart = ferrum.jointplot(df, x="total_bill", y="tip")
    (out_dir / "ferrum.svg").write_text(chart.show_svg(), encoding="utf-8")


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); main(out)
