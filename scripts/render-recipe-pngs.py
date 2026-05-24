"""Render all customization recipe scripts to PNG for documentation.

Imports each recipe module, finds the top-level ``chart`` variable, calls
``show_png(scale=2.0)``, and writes the result to ``docs/site/assets/recipes/``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
RECIPE_DIR = REPO_ROOT / "docs/site/recipes/customization"
OUTPUT_DIR = REPO_ROOT / "docs/site/assets/recipes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_recipe(script_path: Path) -> None:
    """Execute a recipe script and save its ``chart`` variable as PNG."""
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    chart = getattr(mod, "chart", None)
    if chart is None:
        print(f"  SKIP {script_path.name}: no 'chart' variable")
        return

    png_bytes = chart.show_png(scale=2.0)
    out_path = OUTPUT_DIR / f"{script_path.stem}.png"
    out_path.write_bytes(png_bytes)
    print(f"  OK   {out_path.relative_to(REPO_ROOT)} ({len(png_bytes):,} bytes)")


if __name__ == "__main__":
    recipes = sorted(RECIPE_DIR.glob("*.py"))
    print(f"Rendering {len(recipes)} recipes to {OUTPUT_DIR.relative_to(REPO_ROOT)}/")
    errors: list[tuple[str, Exception]] = []
    for r in recipes:
        try:
            run_recipe(r)
        except Exception as exc:
            print(f"  FAIL {r.name}: {exc}")
            errors.append((r.name, exc))

    print()
    if errors:
        print(f"{len(errors)} recipe(s) failed:")
        for name, exc in errors:
            print(f"  {name}: {exc}")
        sys.exit(1)
    else:
        print(f"All {len(recipes)} recipes rendered successfully.")
