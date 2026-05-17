"""Tests for ferrum.color and ferrum.config modules (Phase 12)."""

from __future__ import annotations

import re
import threading

import ferrum


# ---------------------------------------------------------------------------
# ferrum.color tests
# ---------------------------------------------------------------------------

_HEX_RE = re.compile(r"^#[0-9a-f]{6}$")


def _is_hex(s: str) -> bool:
    return bool(_HEX_RE.match(s))


class TestPalette:
    def test_tableau10_returns_10_hex_strings(self):
        colors = ferrum.color.palette("tableau10")
        assert len(colors) == 10
        assert all(_is_hex(c) for c in colors)

    def test_viridis_n5_returns_5_hex_strings(self):
        colors = ferrum.color.palette("viridis", n=5)
        assert len(colors) == 5
        assert all(_is_hex(c) for c in colors)

    def test_categorical_wrap_cyclic(self):
        colors = ferrum.color.palette("okabe_ito", n=16)
        assert len(colors) == 16
        # Should cycle: color[8] == color[0]
        assert colors[8] == colors[0]

    def test_diverging_via_palette(self):
        colors = ferrum.color.palette("rdbu", n=7)
        assert len(colors) == 7
        assert all(_is_hex(c) for c in colors)

    def test_unknown_palette_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown palette"):
            ferrum.color.palette("nonexistent_palette")


class TestToHex:
    def test_float_tuple_red(self):
        assert ferrum.color.to_hex((1.0, 0.0, 0.0)) == "#ff0000"

    def test_int_tuple_red(self):
        assert ferrum.color.to_hex((255, 0, 0)) == "#ff0000"

    def test_float_tuple_green(self):
        assert ferrum.color.to_hex((0.0, 1.0, 0.0)) == "#00ff00"

    def test_int_tuple_blue(self):
        assert ferrum.color.to_hex((0, 0, 255)) == "#0000ff"

    def test_hex_string_passthrough(self):
        assert ferrum.color.to_hex("#abcdef") == "#abcdef"

    def test_mixed_types_detects_float(self):
        # If any component is float, treat all as [0, 1]
        result = ferrum.color.to_hex((0.5, 0.5, 0.5))
        assert _is_hex(result)
        # ~128
        assert result == "#808080"

    def test_invalid_input_raises(self):
        import pytest

        with pytest.raises(ValueError):
            ferrum.color.to_hex("not_a_hex")


class TestSequential:
    def test_viridis_n10(self):
        colors = ferrum.color.sequential("viridis", n=10)
        assert len(colors) == 10
        assert all(_is_hex(c) for c in colors)

    def test_endpoints_match_stops(self):
        colors = ferrum.color.sequential("viridis", n=7)
        # First color should be the first stop
        assert colors[0] == "#440154"
        # Last color should be the last stop
        assert colors[-1] == "#fde725"

    def test_unknown_sequential_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown sequential"):
            ferrum.color.sequential("bogus")


class TestDiverging:
    def test_rdbu_n5(self):
        colors = ferrum.color.diverging("rdbu", n=5)
        assert len(colors) == 5
        assert all(_is_hex(c) for c in colors)

    def test_unknown_diverging_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown diverging"):
            ferrum.color.diverging("bogus")


# ---------------------------------------------------------------------------
# ferrum.config tests
# ---------------------------------------------------------------------------


class TestConfigGet:
    def test_default_width(self):
        ferrum.config.reset()
        assert ferrum.config.get("width") == 640

    def test_default_height(self):
        ferrum.config.reset()
        assert ferrum.config.get("height") == 480

    def test_default_renderer(self):
        ferrum.config.reset()
        assert ferrum.config.get("renderer") == "svg"

    def test_unknown_key_raises(self):
        import pytest

        ferrum.config.reset()
        with pytest.raises(KeyError, match="Unknown config key"):
            ferrum.config.get("nonexistent_key")


class TestConfigSet:
    def test_set_and_get(self):
        ferrum.config.reset()
        ferrum.config.set(width=800)
        assert ferrum.config.get("width") == 800
        ferrum.config.reset()

    def test_set_multiple(self):
        ferrum.config.reset()
        ferrum.config.set(width=1000, height=700)
        assert ferrum.config.get("width") == 1000
        assert ferrum.config.get("height") == 700
        ferrum.config.reset()


class TestConfigDefaults:
    def test_context_manager_scopes(self):
        ferrum.config.reset()
        assert ferrum.config.get("width") == 640
        with ferrum.config.defaults(width=1000):
            assert ferrum.config.get("width") == 1000
        assert ferrum.config.get("width") == 640

    def test_nested_context_managers(self):
        ferrum.config.reset()
        with ferrum.config.defaults(width=800):
            assert ferrum.config.get("width") == 800
            with ferrum.config.defaults(width=1200):
                assert ferrum.config.get("width") == 1200
            assert ferrum.config.get("width") == 800
        assert ferrum.config.get("width") == 640


class TestConfigReset:
    def test_reset_clears_overrides(self):
        ferrum.config.set(width=1200, height=900)
        ferrum.config.reset()
        assert ferrum.config.get("width") == 640
        assert ferrum.config.get("height") == 480


class TestConfigThreadIsolation:
    def test_threads_isolated(self):
        """Config set in one thread does not affect another."""
        ferrum.config.reset()
        results: dict[str, int] = {}
        barrier = threading.Barrier(2)

        def thread_a():
            ferrum.config.set(width=999)
            barrier.wait()
            results["a"] = ferrum.config.get("width")

        def thread_b():
            barrier.wait()
            results["b"] = ferrum.config.get("width")

        t1 = threading.Thread(target=thread_a)
        t2 = threading.Thread(target=thread_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["a"] == 999
        # Thread B should see the default (640) since contextvars are
        # inherited at thread creation time — but thread_b was created
        # before thread_a's set() call, so it sees the reset state.
        assert results["b"] == 640
