"""Theme class, built-in named themes, and default-theme controls.

Pure re-export façade. ``Theme`` is defined in the leaf module
``ferrum.themes._theme`` (which imports nothing from this package); the
builtins live in ``ferrum.themes.builtins`` and the default-theme controls in
``ferrum.themes._defaults``. The one-way dependency chain
``_theme → builtins → _defaults → __init__`` lets this module use plain
top-of-module imports with no statement-ordering tricks.
"""

from __future__ import annotations

from ferrum.themes._theme import Theme
from ferrum.themes import builtins as _builtins
from ferrum.themes._defaults import (
    get_default_theme,
    set_default_theme,
    theme_context,
)

# Re-export the 12 builtins as module attributes
default = _builtins.default
paper_ink = _builtins.paper_ink
slate_citrus = _builtins.slate_citrus
arctic_signal = _builtins.arctic_signal
observable = _builtins.observable
minimal = _builtins.minimal
dark = _builtins.dark
publication = _builtins.publication
economist = _builtins.economist
fivethirtyeight = _builtins.fivethirtyeight
solarized_light = _builtins.solarized_light
solarized_dark = _builtins.solarized_dark

__all__ = [
    "Theme",
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
    "set_default_theme",
    "get_default_theme",
    "theme_context",
]
