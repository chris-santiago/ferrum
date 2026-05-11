#!/usr/bin/env python
"""Rasterize ferrum SVG goldens to PNG for visual inspection.

Usage:
    python scripts/snapshot-goldens.py                    # snapshot ALL goldens
    python scripts/snapshot-goldens.py heatmap_annot      # by stem name
    python scripts/snapshot-goldens.py phase_10/roc_*     # glob pattern
    python scripts/snapshot-goldens.py --clean            # delete tests/snapshots/
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from tests._snapshots import (  # noqa: E402
    DEFAULT_OUT_DIR,
    find_goldens,
    iter_default_roots,
    snapshot_golden,
)


def _match(svg: Path, pattern: str) -> bool:
    """Match against stem, full filename, or any path component glob."""
    if pattern == svg.stem or pattern == svg.name:
        return True
    if fnmatch.fnmatch(svg.stem, pattern) or fnmatch.fnmatch(svg.name, pattern):
        return True
    # Allow "phase_10/roc_*" — match the suffix of the path.
    rel_str = str(svg)
    return fnmatch.fnmatch(rel_str, f"*{pattern}*") or fnmatch.fnmatch(
        rel_str, f"*{pattern}"
    )


def _filter(svgs: list[Path], patterns: list[str]) -> list[Path]:
    if not patterns:
        return svgs
    out: list[Path] = []
    for svg in svgs:
        if any(_match(svg, p) for p in patterns):
            out.append(svg)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "patterns",
        nargs="*",
        help="Optional name or glob filters; default snapshots everything.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete tests/snapshots/ and exit.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory (default: tests/snapshots/).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=96,
        help="Rasterization DPI (default: 96).",
    )
    args = parser.parse_args()

    if args.clean:
        if args.out_dir.exists():
            # Preserve .gitignore so the directory remains version-controlled.
            removed = 0
            for entry in args.out_dir.iterdir():
                if entry.name == ".gitignore":
                    continue
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
                removed += 1
            print(f"removed {removed} entries from {args.out_dir}")
        else:
            print(f"nothing to clean: {args.out_dir} does not exist")
        return 0

    roots = list(iter_default_roots())
    svgs = find_goldens(*roots)
    if not svgs:
        print("no goldens found", file=sys.stderr)
        return 1

    selected = _filter(svgs, args.patterns)
    if not selected:
        print(
            f"no goldens matched pattern(s): {args.patterns}",
            file=sys.stderr,
        )
        return 1

    for svg in selected:
        out_path = snapshot_golden(svg, out_dir=args.out_dir, dpi=args.dpi)
        size_kb = out_path.stat().st_size / 1024
        rel = out_path.relative_to(_REPO_ROOT) if out_path.is_relative_to(_REPO_ROOT) else out_path
        print(f"wrote {rel} ({size_kb:.1f}KB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
