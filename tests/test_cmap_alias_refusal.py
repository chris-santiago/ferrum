"""#240-4a (routed rider): ``resolve_cmap_alias`` refuses an unrecognized
``cmap=``/``scheme=`` name at Python desugar, instead of silently falling
back to Viridis.

Pre-change, an unknown name reached two Rust mark sites that read
``mark_style.group.cmap`` directly with no encoding channel in the loop
(``render/marks/image.rs``'s ``mark_raster``, ``render/marks/polygon.rs``'s
quantitative-color ``mark_hex``) and silently fell back to Viridis with no
diagnostic at all -- ``resolve_continuous_scheme``'s
``RenderWarning::UnknownColorScheme`` (T9, #240-4b) only fires for a
channel-bound scheme, so those two channel-less sites never reached it.
Refusing the name at ``resolve_cmap_alias`` -- the one shared desugar-time
helper every ``cmap=``/``scheme=`` call site already routes through -- makes
both sites unreachable with bad input, per the standing "refuse malformed"
policy.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm
from ferrum.marks._desugar_helpers import resolve_cmap_alias
from ferrum.schemes import continuous_palette


def _points_df() -> pl.DataFrame:
    return pl.DataFrame({"x": [1.0, 2.0, 3.0, 2.5], "y": [4.0, 5.0, 4.5, 4.0]})


# ---------------------------------------------------------------------------
# The shared helper itself
# ---------------------------------------------------------------------------


def test_resolve_cmap_alias_refuses_unknown_scheme_name():
    with pytest.raises(ValueError, match="not recognized"):
        resolve_cmap_alias(scheme="not-a-real-scheme", cmap=None, where="mark_raster")


def test_resolve_cmap_alias_refuses_unknown_cmap_alias_name():
    with pytest.raises(ValueError, match="not recognized"):
        resolve_cmap_alias(scheme=None, cmap="not-a-real-scheme", where="mark_raster")


def test_resolve_cmap_alias_message_names_where_and_available_names():
    with pytest.raises(ValueError) as exc_info:
        resolve_cmap_alias(scheme="not-a-real-scheme", cmap=None, where="mark_raster")
    message = str(exc_info.value)
    assert "mark_raster" in message
    assert "not-a-real-scheme" in message
    assert "viridis" in message  # a registered name, from the "available" list


def test_resolve_cmap_alias_none_stays_none():
    assert resolve_cmap_alias(scheme=None, cmap=None, where="test") is None


def test_resolve_cmap_alias_accepts_every_registered_continuous_name():
    """Desugars identically to before this change for every valid name --
    the refusal only rejects names outside the registered set."""
    for name in continuous_palette.list():
        assert resolve_cmap_alias(scheme=name, cmap=None, where="test") == name
        assert resolve_cmap_alias(scheme=None, cmap=name, where="test") == name


def test_resolve_cmap_alias_case_insensitive_like_rust_resolver():
    # NamedContinuous::from_name lowercases before matching; the Python-side
    # refusal must not be stricter than the Rust resolver it is gating.
    assert resolve_cmap_alias(scheme="Viridis", cmap=None, where="test") == "Viridis"


def test_resolve_cmap_alias_still_refuses_conflicting_alias_pair():
    """Pre-existing conflict refusal (both scheme= and cmap= with different
    values) survives untouched alongside the new name refusal."""
    with pytest.raises(ValueError, match="pass either"):
        resolve_cmap_alias(scheme="viridis", cmap="plasma", where="mark_raster")


# ---------------------------------------------------------------------------
# The two channel-less mark call sites the rider names
# ---------------------------------------------------------------------------


def test_mark_raster_refuses_unknown_cmap_at_call_time():
    with pytest.raises(ValueError, match="not recognized"):
        fm.Chart(_points_df()).mark_raster(cmap="not-a-real-scheme")


def test_mark_hex_refuses_unknown_scheme_at_call_time():
    with pytest.raises(ValueError, match="not recognized"):
        fm.Chart(_points_df()).mark_hex(scheme="not-a-real-scheme")


def test_mark_contour_refuses_unknown_cmap_at_call_time():
    with pytest.raises(ValueError, match="not recognized"):
        fm.Chart(_points_df()).mark_contour(cmap="not-a-real-scheme")


# ---------------------------------------------------------------------------
# The two figure-level consumers
# ---------------------------------------------------------------------------


def test_heatmap_refuses_unknown_cmap():
    df = pl.DataFrame({"row": ["a", "b"], "v1": [1.0, 2.0], "v2": [3.0, 4.0]})
    with pytest.raises(ValueError, match="not recognized"):
        fm.heatmap(df, cmap="not-a-real-scheme")


def test_clustermap_refuses_unknown_scheme():
    df = pl.DataFrame(
        {
            "row": ["a", "b", "c"],
            "v1": [1.0, 2.0, 3.0],
            "v2": [3.0, 1.0, 2.0],
            "v3": [2.0, 3.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="not recognized"):
        fm.clustermap(df, scheme="not-a-real-scheme")


def test_render_config_refuses_unknown_raster_scheme():
    from ferrum.render_config import RenderConfig

    with pytest.raises(ValueError, match="not recognized"):
        RenderConfig(raster_scheme="not-a-real-scheme")
