"""Tests for G-D7 (headline): stacked radial bars via mark_bar + CoordPolar.

Root cause: ``Radius._honored_kwargs`` was ``frozenset(["type"])``, so
``Radius(stack=...)`` fired a spurious deprecation warning and bool ``True``
was not normalized to ``"zero"`` before reaching the PyO3 boundary (which
expects ``Option<String>``).

Fix: add ``"stack"`` to ``Radius._honored_kwargs`` and normalize
``stack=True`` → ``"zero"`` / ``stack=False`` → ``"false"`` in
``Radius._validate``, matching the pattern in ``Y._validate``.

The Rust stacking path already existed: under ``CoordPolar``, ``radius`` is
remapped to ``y`` during ``_resolve_polar_remapping``; ``apply_stack``
(position.rs) reads ``encoding.y.stack`` and populates
``__stack_y_base__``; ``bar.rs build_polar`` reads that base as ``inner_r``
so wedges stack outward.  The only missing piece was the Python-side
pass-through of the ``stack`` directive.
"""

from __future__ import annotations

import json
import re
import warnings

import polars as pl
import pytest

import ferrum as fm


# ---------------------------------------------------------------------------
# SVG geometry helpers (local copies — no shared test-utility import)
# ---------------------------------------------------------------------------


def _filled_arc_paths(svg: str) -> list[str]:
    """Return ``d=`` values for <path> elements with arc commands and non-none fill."""
    result = []
    for attrs in re.findall(r"<path([^>]+)>", svg):
        d_m = re.search(r'd="([^"]+)"', attrs)
        if d_m is None:
            continue
        d = d_m.group(1)
        if "A" not in d:
            continue
        fill_m = re.search(r'fill="([^"]+)"', attrs)
        if fill_m is None or fill_m.group(1) == "none":
            continue
        result.append(d)
    return result


def _inner_outer_radii(path_d: str) -> tuple[float, float]:
    """Return ``(inner_r, outer_r)`` from a wedge path.

    A solid wedge has one arc (outer only); inner_r is 0.
    An annular wedge has two arcs: first = outer, second = inner.
    """
    arc_radii = [float(m.group(1)) for m in re.finditer(r"A\s*([\d.]+)", path_d)]
    if not arc_radii:
        raise ValueError(f"no arc commands in path: {path_d[:80]!r}")
    if len(arc_radii) == 1:
        return 0.0, arc_radii[0]
    return arc_radii[1], arc_radii[0]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wind_rose_df() -> pl.DataFrame:
    """4 directions × 3 categories, values 10 / 20 / 30.

    Correct stacking (data units): A → 0..10, B → 10..30, C → 30..60.
    """
    directions = ["N", "E", "S", "W"] * 3
    categories = ["A"] * 4 + ["B"] * 4 + ["C"] * 4
    values = [10.0] * 4 + [20.0] * 4 + [30.0] * 4
    return pl.DataFrame({"direction": directions, "category": categories, "value": values})


@pytest.fixture
def cartesian_stack_df() -> pl.DataFrame:
    """Standard cartesian stacked-bar dataset (back-compat fixture)."""
    return pl.DataFrame(
        {
            "cat": ["A", "A", "B", "B"],
            "grp": ["X", "Y", "X", "Y"],
            "val": [10.0, 20.0, 30.0, 40.0],
        }
    )


# ---------------------------------------------------------------------------
# 1. Radius._honored_kwargs and warn suppression
# ---------------------------------------------------------------------------


class TestRadiusStackHonoredKwarg:
    """``stack`` must be in ``Radius._honored_kwargs`` so no warning fires."""

    def test_stack_in_honored_kwargs(self) -> None:
        """``"stack"`` must be present in ``Radius._honored_kwargs``."""
        assert "stack" in fm.Radius._honored_kwargs, (
            "'stack' is missing from Radius._honored_kwargs — "
            "Radius(stack=...) fires a spurious deprecation warning and the "
            "directive may be silently ignored."
        )

    def test_radius_stack_zero_no_warning(self) -> None:
        """``Radius(field, stack='zero')`` must not emit a UserWarning."""
        from ferrum._warn import reset_warnings

        reset_warnings()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fm.Radius("val", stack="zero")
        stack_warns = [w for w in caught if "stack" in str(w.message).lower()]
        assert not stack_warns, f"Radius(stack='zero') emitted unexpected warning(s): {stack_warns}"

    def test_radius_stack_true_no_warning(self) -> None:
        """``Radius(field, stack=True)`` must not emit a UserWarning."""
        from ferrum._warn import reset_warnings

        reset_warnings()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fm.Radius("val", stack=True)
        stack_warns = [w for w in caught if "stack" in str(w.message).lower()]
        assert not stack_warns, f"Radius(stack=True) emitted unexpected warning(s): {stack_warns}"


# ---------------------------------------------------------------------------
# 2. Serialization: stack directive reaches EncodingSpec / spec JSON
# ---------------------------------------------------------------------------


class TestRadiusStackSerialization:
    """The ``stack`` directive must survive the full Python→Rust serialization."""

    def test_radius_stack_zero_in_spec_dict(self) -> None:
        """``Radius('v', stack='zero').to_encoding_spec_dict()`` must include ``stack``."""
        d = fm.Radius("val", stack="zero").to_encoding_spec_dict()
        assert d.get("stack") == "zero", (
            f"to_encoding_spec_dict() returned {d!r}; expected 'stack': 'zero'."
        )

    def test_radius_stack_true_normalized_to_zero(self) -> None:
        """``Radius('v', stack=True)`` must normalize to ``stack='zero'``."""
        r = fm.Radius("val", stack=True)
        d = r.to_encoding_spec_dict()
        assert d.get("stack") == "zero", (
            f"Radius(stack=True) should normalize to 'zero'; got {d.get('stack')!r}. "
            "Rust EncodingSpec.stack is Option<String> — bool True would TypeError."
        )

    def test_radius_stack_normalize_in_spec_dict(self) -> None:
        """``Radius('v', stack='normalize')`` must pass 'normalize' to spec dict."""
        d = fm.Radius("val", stack="normalize").to_encoding_spec_dict()
        assert d.get("stack") == "normalize", (
            f"Expected 'normalize' in spec dict; got {d.get('stack')!r}."
        )

    def test_stacked_coxcomb_spec_json_carries_stack_on_y(self, wind_rose_df: pl.DataFrame) -> None:
        """After polar remapping, ``radius`` → ``y``; spec JSON must carry ``stack`` on y.

        This confirms the directive reaches Rust's ``apply_stack`` trigger
        path (``encoding.y.stack``).
        """
        chart = (
            fm.Chart(wind_rose_df)
            .mark_bar()
            .encode(
                theta="direction:N",
                radius=fm.Radius("value", stack="zero"),
                color="category:N",
            )
            .coord(fm.CoordPolar(theta="x"))
        )
        spec = chart.to_spec()
        parsed = json.loads(spec.to_json())
        y_enc = parsed.get("encoding", {}).get("y", {})
        assert y_enc.get("stack") == "zero", (
            f"After polar remapping (radius→y), spec.encoding.y must carry "
            f"'stack': 'zero', but got y_enc={y_enc!r}. "
            "The directive never reached apply_stack."
        )

    def test_stacked_coxcomb_via_position_stack_spec_has_position(
        self, wind_rose_df: pl.DataFrame
    ) -> None:
        """``mark_bar(position=Stack())`` must include the position in the spec."""
        chart = (
            fm.Chart(wind_rose_df)
            .mark_bar(position=fm.Stack())
            .encode(x="direction:N", y="value:Q", color="category:N")
            .coord(fm.CoordPolar(theta="x"))
        )
        spec = chart.to_spec()
        parsed = json.loads(spec.to_json())
        pos = parsed.get("position")
        assert pos is not None and pos.get("type") == "stack", (
            f"Expected position.type='stack' in spec; got {pos!r}."
        )


# ---------------------------------------------------------------------------
# 3. Behavioral: stacked wedges accumulate outward
# ---------------------------------------------------------------------------


class TestRadialStackGeometry:
    """Stacked polar segments must render as wedges with outward-accumulating radii."""

    def test_position_stack_produces_arc_wedges(self, wind_rose_df: pl.DataFrame) -> None:
        """``mark_bar(position=Stack())`` + CoordPolar must render filled arc paths."""
        svg = (
            fm.Chart(wind_rose_df)
            .mark_bar(position=fm.Stack())
            .encode(x="direction:N", y="value:Q", color="category:N")
            .coord(fm.CoordPolar(theta="x"))
            .to_svg()
        )
        paths = _filled_arc_paths(svg)
        assert paths, (
            "No filled arc wedge paths in stacked polar bar SVG. "
            "mark_bar under CoordPolar must render arc wedge paths, not rects."
        )

    def test_encoding_stack_produces_arc_wedges(self, wind_rose_df: pl.DataFrame) -> None:
        """``Radius('value', stack='zero')`` must also render filled arc paths."""
        svg = (
            fm.Chart(wind_rose_df)
            .mark_bar()
            .encode(
                theta="direction:N",
                radius=fm.Radius("value", stack="zero"),
                color="category:N",
            )
            .coord(fm.CoordPolar(theta="x"))
            .to_svg()
        )
        paths = _filled_arc_paths(svg)
        assert paths, "No filled arc wedge paths in Radius(stack='zero') polar bar SVG."

    def test_position_stack_segments_accumulate_outward(self, wind_rose_df: pl.DataFrame) -> None:
        """Stacked segments must have non-zero inner radii for outer groups.

        Dataset: A(10), B(20), C(30).  Correct radial accumulation:
          A: inner=0, outer=r1
          B: inner=r1, outer=r1+r2
          C: inner=r1+r2, outer=r1+r2+r3
        At least one segment must have inner_r > 1 px.
        """
        svg = (
            fm.Chart(wind_rose_df)
            .mark_bar(position=fm.Stack())
            .encode(x="direction:N", y="value:Q", color="category:N")
            .coord(fm.CoordPolar(theta="x"))
            .to_svg()
        )
        paths = _filled_arc_paths(svg)
        assert paths, "no filled arc paths to inspect"

        inner_radii = []
        for d in paths:
            try:
                inner_r, _ = _inner_outer_radii(d)
                inner_radii.append(inner_r)
            except ValueError:
                continue

        non_zero = [r for r in inner_radii if r > 1.0]
        assert non_zero, (
            f"All stacked polar segments have inner_r <= 1 px: {inner_radii}. "
            "Stacked wedges must accumulate outward — segment B's inner radius "
            "must equal segment A's outer radius."
        )

    def test_encoding_stack_segments_accumulate_outward(self, wind_rose_df: pl.DataFrame) -> None:
        """Same outward-accumulation check via the encoding-level stack path."""
        svg = (
            fm.Chart(wind_rose_df)
            .mark_bar()
            .encode(
                theta="direction:N",
                radius=fm.Radius("value", stack="zero"),
                color="category:N",
            )
            .coord(fm.CoordPolar(theta="x"))
            .to_svg()
        )
        paths = _filled_arc_paths(svg)
        assert paths, "no filled arc paths to inspect"

        inner_radii = []
        for d in paths:
            try:
                inner_r, _ = _inner_outer_radii(d)
                inner_radii.append(inner_r)
            except ValueError:
                continue

        non_zero = [r for r in inner_radii if r > 1.0]
        assert non_zero, f"Radius(stack='zero') segments all have inner_r <= 1 px: {inner_radii}."

    def test_stacked_differs_from_unstacked(self, wind_rose_df: pl.DataFrame) -> None:
        """Stacked render must produce different radial extents than unstacked.

        This proves accumulation is real, not an accident of geometry.
        Compare the maximum inner_r: unstacked must be 0 for all segments;
        stacked must have at least one inner_r > 0.
        """
        stacked_svg = (
            fm.Chart(wind_rose_df)
            .mark_bar(position=fm.Stack())
            .encode(x="direction:N", y="value:Q", color="category:N")
            .coord(fm.CoordPolar(theta="x"))
            .to_svg()
        )
        unstacked_svg = (
            fm.Chart(wind_rose_df)
            .mark_bar()
            .encode(x="direction:N", y="value:Q", color="category:N")
            .coord(fm.CoordPolar(theta="x"))
            .to_svg()
        )

        def _max_inner(svg: str) -> float:
            paths = _filled_arc_paths(svg)
            inner_rs = []
            for d in paths:
                try:
                    inner_r, _ = _inner_outer_radii(d)
                    inner_rs.append(inner_r)
                except ValueError:
                    continue
            return max(inner_rs) if inner_rs else 0.0

        stacked_max_inner = _max_inner(stacked_svg)
        unstacked_max_inner = _max_inner(unstacked_svg)

        assert stacked_max_inner > 1.0, (
            f"Stacked polar render has max inner_r={stacked_max_inner:.2f}; "
            "expected > 1 px to prove outward accumulation."
        )
        assert stacked_max_inner > unstacked_max_inner, (
            f"Stacked max_inner_r={stacked_max_inner:.2f} is not greater than "
            f"unstacked max_inner_r={unstacked_max_inner:.2f}; "
            "stacking must produce larger inner radii."
        )

    def test_single_direction_segments_are_contiguous(self, wind_rose_df: pl.DataFrame) -> None:
        """For a single direction, adjacent stacked segments must share a boundary.

        Segment i's outer_r must equal segment i+1's inner_r within 2 px.
        """
        single = wind_rose_df.filter(pl.col("direction") == "N")
        svg = (
            fm.Chart(single)
            .mark_bar(position=fm.Stack())
            .encode(x="direction:N", y="value:Q", color="category:N")
            .coord(fm.CoordPolar(theta="x"))
            .to_svg()
        )
        paths = _filled_arc_paths(svg)
        assert paths, "no arc paths for single-direction stacked chart"

        segments: list[tuple[float, float]] = []
        for d in paths:
            try:
                inner_r, outer_r = _inner_outer_radii(d)
                segments.append((inner_r, outer_r))
            except ValueError:
                continue

        assert len(segments) >= 2, f"Expected at least 2 stacked segments, got {len(segments)}"
        segments.sort()

        for i in range(len(segments) - 1):
            _, outer = segments[i]
            inner_next, _ = segments[i + 1]
            assert abs(outer - inner_next) < 2.0, (
                f"Segment {i} outer_r={outer:.2f} != segment {i + 1} "
                f"inner_r={inner_next:.2f}: segments are not contiguous "
                "(overlap bug — all segments start at r=0)."
            )


# ---------------------------------------------------------------------------
# 4. Radius validation: invalid stack values
# ---------------------------------------------------------------------------


class TestRadiusStackValidation:
    """Invalid ``stack=`` values must raise ``ValueError``."""

    def test_radius_stack_invalid_string_raises(self) -> None:
        """``Radius(field, stack='bogus')`` must raise ``ValueError``."""
        with pytest.raises(ValueError, match="stack"):
            fm.Radius("val", stack="bogus")

    def test_radius_stack_false_accepted(self) -> None:
        """``Radius(field, stack=False)`` must normalize to the string ``"false"``."""
        r = fm.Radius("val", stack=False)
        d = r.to_encoding_spec_dict()
        stack_val = d.get("stack")
        assert stack_val == "false", (
            f"stack=False must normalize to the string 'false' so Rust's "
            f"Option<String> parser returns None (no-stack); got {stack_val!r}"
        )


# ---------------------------------------------------------------------------
# 5. Back-compat: Theta and non-polar stacking unchanged
# ---------------------------------------------------------------------------


class TestBackCompat:
    """Existing Theta and cartesian bar stacking must not regress."""

    def test_theta_stack_still_accepted(self) -> None:
        """``Theta('v', stack=True)`` must remain accepted without error."""
        th = fm.Theta("val", stack=True)
        assert "stack" in th._honored_kwargs

    def test_theta_stack_kwarg_present(self) -> None:
        """``Theta('v', stack=True)`` must store the stack kwarg."""
        th = fm.Theta("val", stack=True)
        assert th._kwargs.get("stack") is not None

    def test_cartesian_stack_still_works(self, cartesian_stack_df: pl.DataFrame) -> None:
        """``mark_bar(position=Stack())`` on a Cartesian chart must render without error."""
        svg = (
            fm.Chart(cartesian_stack_df)
            .mark_bar(position=fm.Stack())
            .encode(x="cat:N", y="val:Q", color="grp:N")
            .to_svg()
        )
        assert "<svg" in svg, "Cartesian stacked bar did not render to SVG"

    def test_y_stack_zero_still_works(self, cartesian_stack_df: pl.DataFrame) -> None:
        """``Y('val', stack='zero')`` on a Cartesian chart must render without error."""
        svg = (
            fm.Chart(cartesian_stack_df)
            .mark_bar()
            .encode(x="cat:N", y=fm.Y("val", stack="zero"), color="grp:N")
            .to_svg()
        )
        assert "<svg" in svg, "Cartesian Y(stack='zero') bar did not render to SVG"

    def test_radius_without_stack_unchanged(self, wind_rose_df: pl.DataFrame) -> None:
        """``Radius('value')`` without ``stack`` must render a non-stacked chart."""
        svg = (
            fm.Chart(wind_rose_df)
            .mark_arc()
            .encode(
                theta="direction:N",
                radius=fm.Radius("value"),
                color="category:N",
            )
            .coord(fm.CoordPolar(theta="x"))
            .to_svg()
        )
        assert "<svg" in svg, "Non-stacked Radius chart did not render"


# ---------------------------------------------------------------------------
# 6. Sibling-channel normalization: X, Y, Theta all normalize stack bools
# ---------------------------------------------------------------------------


class TestSiblingStackNormalization:
    """X, Y, and Theta must normalize bool stack values like Radius does.

    Before the fix, bool True/False passed through to Rust's Option<String>
    stack field and caused ``TypeError: argument 'stack': 'bool' object is
    not an instance of 'str'`` at the PyO3 boundary.
    """

    def test_y_stack_true_normalizes_to_zero(self) -> None:
        """``Y('v', stack=True)`` must normalize to ``stack='zero'``."""
        d = fm.Y("val", stack=True).to_encoding_spec_dict()
        assert d.get("stack") == "zero", (
            f"Y(stack=True) must normalize to 'zero'; got {d.get('stack')!r}"
        )

    def test_y_stack_false_normalizes_to_string(self) -> None:
        """``Y('v', stack=False)`` must normalize to the string ``'false'``."""
        d = fm.Y("val", stack=False).to_encoding_spec_dict()
        assert d.get("stack") == "false", (
            f"Y(stack=False) must normalize to 'false'; got {d.get('stack')!r}"
        )

    def test_y_stack_normalize_string_passes_through(self) -> None:
        """``Y('v', stack='normalize')`` must pass through unchanged."""
        d = fm.Y("val", stack="normalize").to_encoding_spec_dict()
        assert d.get("stack") == "normalize"

    def test_y_stack_bogus_raises_value_error(self) -> None:
        """``Y('v', stack='bogus')`` must raise ``ValueError``."""
        with pytest.raises(ValueError, match="stack"):
            fm.Y("val", stack="bogus")

    def test_theta_stack_true_normalizes_to_zero(self) -> None:
        """``Theta('v', stack=True)`` must normalize to ``stack='zero'``."""
        d = fm.Theta("val", stack=True).to_encoding_spec_dict()
        assert d.get("stack") == "zero", (
            f"Theta(stack=True) must normalize to 'zero'; got {d.get('stack')!r}"
        )

    def test_theta_stack_false_normalizes_to_string(self) -> None:
        """``Theta('v', stack=False)`` must normalize to the string ``'false'``."""
        d = fm.Theta("val", stack=False).to_encoding_spec_dict()
        assert d.get("stack") == "false", (
            f"Theta(stack=False) must normalize to 'false'; got {d.get('stack')!r}"
        )

    def test_x_stack_true_normalizes_to_zero(self) -> None:
        """``X('v', stack=True)`` must normalize to ``stack='zero'``."""
        d = fm.X("val", stack=True).to_encoding_spec_dict()
        assert d.get("stack") == "zero", (
            f"X(stack=True) must normalize to 'zero'; got {d.get('stack')!r}"
        )

    def test_x_stack_false_normalizes_to_string(self) -> None:
        """``X('v', stack=False)`` must normalize to the string ``'false'``."""
        d = fm.X("val", stack=False).to_encoding_spec_dict()
        assert d.get("stack") == "false", (
            f"X(stack=False) must normalize to 'false'; got {d.get('stack')!r}"
        )

    def test_y_stack_true_renders_stacked_bar(self, cartesian_stack_df: pl.DataFrame) -> None:
        """``Y('val', stack=True)`` must not crash at the PyO3 boundary.

        This is the regression test for the latent crash: bool True was passed
        to Rust's ``Option<String>`` stack field, causing a TypeError.
        """
        svg = (
            fm.Chart(cartesian_stack_df)
            .mark_bar()
            .encode(
                x="cat:N",
                y=fm.Y("val", stack=True),
                color="grp:N",
            )
            .to_svg()
        )
        assert "<svg" in svg, "Y(stack=True) stacked bar did not render to SVG"

    def test_theta_stack_true_renders_pie(self, cartesian_stack_df: pl.DataFrame) -> None:
        """``Theta('val', stack=True)`` must not crash at the PyO3 boundary."""
        svg = (
            fm.Chart(cartesian_stack_df)
            .mark_arc()
            .encode(
                theta=fm.Theta("val", stack=True),
                color="grp:N",
            )
            .coord(fm.CoordPolar(theta="x"))
            .to_svg()
        )
        assert "<svg" in svg, "Theta(stack=True) arc did not render to SVG"

    def test_x_stack_true_renders_bar(self, cartesian_stack_df: pl.DataFrame) -> None:
        """``X('val', stack=True)`` must not crash at the PyO3 boundary."""
        svg = (
            fm.Chart(cartesian_stack_df)
            .mark_bar()
            .encode(
                x=fm.X("val", stack=True),
                y="cat:N",
                color="grp:N",
            )
            .to_svg()
        )
        assert "<svg" in svg, "X(stack=True) horizontal stacked bar did not render to SVG"


# ---------------------------------------------------------------------------
# 7. _normalize_stack type guard: non-bool, non-str raises TypeError (not AssertionError)
# ---------------------------------------------------------------------------


class TestNormalizeStackTypeGuard:
    """Non-bool, non-str stack= values must raise TypeError with a useful message.

    Before the fix, ``_normalize_stack`` used a bare ``assert isinstance(value, str)``
    which raised ``AssertionError`` on bad types and was silently stripped under
    ``python -O``, falling through to ``value.lower()`` → ``AttributeError``.
    """

    def test_radius_stack_int_raises_type_error(self) -> None:
        """``Radius('v', stack=1)`` must raise TypeError (not AssertionError)."""
        with pytest.raises(TypeError, match="stack"):
            fm.Radius("val", stack=1)

    def test_radius_stack_int_message_names_channel(self) -> None:
        """TypeError message must name the channel and the bad value."""
        with pytest.raises(TypeError, match="Radius"):
            fm.Radius("val", stack=1)

    def test_y_stack_list_raises_type_error(self) -> None:
        """``Y('v', stack=[])`` must raise TypeError (not AssertionError)."""
        with pytest.raises(TypeError, match="stack"):
            fm.Y("val", stack=[])

    def test_y_stack_list_message_names_channel(self) -> None:
        """TypeError message must name the channel."""
        with pytest.raises(TypeError, match="Y"):
            fm.Y("val", stack=[])

    def test_theta_stack_none_obj_raises_type_error(self) -> None:
        """``Theta('v', stack=42)`` must raise TypeError."""
        with pytest.raises(TypeError, match="stack"):
            fm.Theta("val", stack=42)

    def test_x_stack_dict_raises_type_error(self) -> None:
        """``X('v', stack={})`` must raise TypeError."""
        with pytest.raises(TypeError, match="stack"):
            fm.X("val", stack={})

    def test_error_is_not_assert_error(self) -> None:
        """The raised exception must not be AssertionError."""
        exc = None
        try:
            fm.Radius("val", stack=1)
        except Exception as e:
            exc = e
        assert exc is not None, "Expected an exception but none was raised"
        assert not isinstance(exc, AssertionError), (
            f"Got AssertionError instead of TypeError; "
            "bare assert was not replaced with explicit raise."
        )
