"""Consistency guards for the built-in themes and the themes package layout.

Two findings from the 2026-06-21 cohesion audit are guarded here:

THEME-06 (import-cycle break)
    ``Theme`` now lives in the leaf module ``ferrum.themes._theme``, which
    imports nothing from the ``ferrum.themes`` package. These tests assert the
    leaf imports standalone and that ``Theme`` is the *same* object via every
    public path, so a regression that reintroduces the cycle (or re-homes
    ``Theme``) fails loudly.

THEME-04 (built-in key-set drift)
    There was no test signal if a built-in theme grew a typo'd key or silently
    dropped a color key (a "half-themed" identity). These tests pin every
    built-in's exact key-set as a characterization and assert every key is a
    valid Rust theme key. Derivation (``paper_ink.update(**deltas)``) was *not*
    used: the 12 builtins are already minimal-delta from the Rust
    ``ThemeInputs::default()`` and are opposite visual identities, so a uniform
    key set would force value changes (golden-affecting). The honest guard is a
    characterization + validity check, not a "every theme sets all identity
    keys" rule -- the full themes legitimately vary.
"""

from __future__ import annotations

import ast
import inspect

import pytest

import ferrum.themes._theme as _theme_mod
from ferrum._core import theme_known_keys
from ferrum.themes import builtins
from ferrum.themes.builtins import IDENTITY_KEYS

# -- The 12 built-in themes, by name ------------------------------------------
BUILTIN_NAMES = [
    "default",
    "paper_ink",
    "slate_citrus",
    "arctic_signal",
    "observable",
    "minimal",
    "dark",
    "publication",
    "economist",
    "fivethirtyeight",
    "solarized_light",
    "solarized_dark",
]

# -- Characterization: the EXACT key-set each built-in sets today --------------
# Pinned so an accidental add/remove (a typo, a half-applied identity edit)
# fails loudly. Update this table deliberately when a theme's keys change on
# purpose. ``default`` is empty (all Rust defaults); ``minimal`` sets only
# structural keys (it is not a color identity).
EXPECTED_KEYS: dict[str, frozenset[str]] = {
    "default": frozenset(),
    "minimal": frozenset(
        {
            "grid",
            "axis_line",
            "tick_size",
            "padding",
            "label_color",
            "sequential_scheme",
            "diverging_scheme",
        }
    ),
    "paper_ink": frozenset(IDENTITY_KEYS),
    "slate_citrus": frozenset(IDENTITY_KEYS),
    "arctic_signal": frozenset(IDENTITY_KEYS),
    "observable": frozenset(IDENTITY_KEYS),
    "dark": frozenset(IDENTITY_KEYS | {"grid_width"}),
    "publication": frozenset(
        {
            "background",
            "grid",
            "axis_line_color",
            "axis_line_width",
            "tick_color",
            "font_color",
            "label_color",
            "title_color",
            "title_font_weight",
            "title_anchor",
            "mark_color",
            "color_scheme",
            "sequential_scheme",
            "diverging_scheme",
            "point_size",
            "reference_line_color",
        }
    ),
    "economist": frozenset(
        {
            "background",
            "font_color",
            "title_color",
            "title_font_weight",
            "title_anchor",
            "axis_line",
            "grid_color",
            "grid_width",
            "mark_color",
            "color_scheme",
            "sequential_scheme",
            "diverging_scheme",
            "strip_background_color",
        }
    ),
    "fivethirtyeight": frozenset(
        {
            "background",
            "font_color",
            "label_color",
            "axis_line",
            "tick_color",
            "grid_color",
            "grid_width",
            "mark_color",
            "color_scheme",
            "sequential_scheme",
            "diverging_scheme",
            "title_font_weight",
            "title_anchor",
        }
    ),
    "solarized_light": frozenset(
        {
            "background",
            "font_color",
            "label_color",
            "title_color",
            "title_font_weight",
            "grid_color",
            "grid_width",
            "axis_line_color",
            "tick_color",
            "mark_color",
            "color_scheme",
            "sequential_scheme",
            "diverging_scheme",
        }
    ),
    "solarized_dark": frozenset(
        {
            "background",
            "font_color",
            "label_color",
            "title_color",
            "title_font_weight",
            "grid_color",
            "grid_width",
            "axis_line_color",
            "tick_color",
            "mark_color",
            "color_scheme",
            "sequential_scheme",
            "diverging_scheme",
            "strip_background_color",
        }
    ),
}


# -- THEME-06: leaf-module isolation + Theme identity -------------------------


def test_theme_leaf_module_imports_standalone():
    # The leaf must import with nothing from ferrum.themes already loaded.
    import importlib

    mod = importlib.import_module("ferrum.themes._theme")
    assert hasattr(mod, "Theme")


def test_theme_leaf_imports_nothing_from_themes_package():
    src = inspect.getsource(_theme_mod)
    tree = ast.parse(src)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("ferrum.themes"):
                offenders.append(node.module)
        elif isinstance(node, ast.Import):
            offenders += [n.name for n in node.names if n.name.startswith("ferrum.themes")]
    assert not offenders, f"leaf module imports from ferrum.themes: {offenders}"


def test_theme_is_same_object_via_every_public_path():
    import ferrum as fm
    from ferrum.themes import Theme as theme_from_package
    from ferrum.themes._theme import Theme as theme_from_leaf

    assert fm.Theme is theme_from_package
    assert fm.Theme is theme_from_leaf
    assert fm.Theme.__module__ == "ferrum.themes._theme"


def test_no_noqa_e402_in_themes_init():
    import ferrum.themes

    src = inspect.getsource(ferrum.themes)
    assert "noqa: E402" not in src, "themes/__init__.py should have no E402 suppressions"


# -- THEME-04: built-in key-set validity + characterization -------------------


def test_identity_keys_subset_of_known_keys():
    known = set(theme_known_keys())
    assert IDENTITY_KEYS <= known, sorted(IDENTITY_KEYS - known)


@pytest.mark.parametrize("name", BUILTIN_NAMES)
def test_builtin_keys_are_valid_theme_keys(name):
    """Every key a built-in sets must be a real Rust theme key (catches typos)."""
    known = set(theme_known_keys())
    theme = getattr(builtins, name)
    stray = set(theme._props) - known
    assert not stray, f"{name} sets unknown key(s): {sorted(stray)}"


@pytest.mark.parametrize("name", BUILTIN_NAMES)
def test_builtin_key_set_matches_characterization(name):
    """Pin each built-in's exact key-set so an accidental add/remove fails."""
    theme = getattr(builtins, name)
    assert set(theme._props) == EXPECTED_KEYS[name], (
        f"{name} key-set drifted from the characterization in EXPECTED_KEYS; "
        f"got {sorted(theme._props)}, expected {sorted(EXPECTED_KEYS[name])}. "
        f"If this change is intentional, update EXPECTED_KEYS."
    )


def test_default_and_minimal_are_not_color_identities():
    """``default`` (empty) and ``minimal`` (structural) set no full color identity."""
    assert builtins.default._props == {}
    assert not (set(builtins.minimal._props) & {"background", "mark_color", "color_scheme"})
