#!/usr/bin/env python3
"""Embed the Inter web font into ``ferrum-interactive.css`` as a data-URI ``@font-face``.

The interactive/anywidget overlay renders all chart text (titles, axis titles,
legend text, tick labels) as SVG ``<text>`` with ``font-family: Inter``, relying
on an ``@font-face`` that the widget injects. That face must cover every weight
the themes request — notably the semibold (600) used for titles — or the browser
synthesises faux-bold and the text renders blurry (GH #80).

This script (re)generates that single ``@font-face`` block from the bundled
**Inter variable** web font, declaring the full ``font-weight: 100 900`` range so
weight-600 text matches real glyphs instead of being synthesised. The variable
woff2 is smaller than the former single-weight Regular ttf, so the interactive
payload shrinks even though it now covers every weight.

Source of truth for the font bytes:
``crates/ferrum-core/assets/fonts/InterVariable.woff2`` (SIL OFL 1.1; see the
sibling ``Inter-OFL.txt``). The same variable family, as ``InterVariable.ttf``,
backs the Rust text-metrics path (``crates/ferrum-core/src/render/font.rs``), so
metrics and rendering agree on one font.

Usage::

    unset CONDA_PREFIX && uv run --no-sync python scripts/embed_interactive_font.py         # rewrite the CSS
    unset CONDA_PREFIX && uv run --no-sync python scripts/embed_interactive_font.py --check  # verify, exit 1 if stale
"""

from __future__ import annotations

import argparse
import base64
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WOFF2 = REPO / "crates" / "ferrum-core" / "assets" / "fonts" / "InterVariable.woff2"
CSS = REPO / "src" / "ferrum" / "_wasm" / "ferrum-interactive.css"

# Matches the single Inter @font-face block the CSS carries (payload-agnostic).
_FONT_FACE_RE = re.compile(r'@font-face\{font-family:"Inter";[^}]*\}')


def _build_font_face() -> str:
    """Return the canonical variable-font ``@font-face`` block as a string."""
    b64 = base64.b64encode(WOFF2.read_bytes()).decode("ascii")
    return (
        '@font-face{font-family:"Inter";font-style:normal;'
        "font-weight:100 900;font-display:swap;"
        f'src:url("data:font/woff2;base64,{b64}") format("woff2");}}'
    )


def _rewrite() -> str:
    """Return the CSS text with its ``@font-face`` replaced by the current font."""
    css = CSS.read_text()
    matches = _FONT_FACE_RE.findall(css)
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one Inter @font-face in {CSS.name}, found {len(matches)}"
        )
    return _FONT_FACE_RE.sub(lambda _m: _build_font_face(), css, count=1)


def main() -> int:
    """Rewrite (or ``--check``) the interactive CSS @font-face from the woff2."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify the CSS already embeds the current font; exit 1 if stale",
    )
    args = ap.parse_args()

    new_css = _rewrite()
    if args.check:
        if new_css != CSS.read_text():
            print(f"STALE: {CSS} does not embed the current {WOFF2.name}", file=sys.stderr)
            return 1
        print(f"OK: {CSS.name} embeds the current {WOFF2.name}")
        return 0

    CSS.write_text(new_css)
    print(f"Wrote {CSS} with {WOFF2.name} ({WOFF2.stat().st_size} bytes, weight 100–900)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
