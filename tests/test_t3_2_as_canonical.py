"""Regression tests for T3.2 — canonical ``as_`` kwarg across transform_* wrappers.

Pins:
- Every transform_* wrapper that accepts an output-column name uses the kwarg
  ``as_`` (not ``as_field``, not ``output``, etc.).
- Wire output is byte-identical: the emitted dict keys are unchanged (e.g.
  transform_calculate still emits ``"as_field"``, not ``"as_"``).
- Invalid ``as_`` shapes raise ``ValueError`` consistently across all wrappers
  (SEAM-03 validation parity).
- The dict-list wrappers (aggregate, join_aggregate, window) pass inner-dict
  ``"as"`` keys verbatim — that is the Vega wire form, not drift.

Findings ENC-05, SEAM-03, D-ASNAME-1.
"""

from __future__ import annotations

import pytest

from ferrum.transforms import (
    transform_bin,
    transform_calculate,
    transform_density,
    transform_flatten,
    transform_fold,
    transform_loess,
    transform_regression,
    transform_stack,
    transform_timeunit,
)


# ---------------------------------------------------------------------------
# 1. Canonical kwarg name: ``as_`` accepted by keyword on every wrapper
# ---------------------------------------------------------------------------


class TestAsKwargIsCanonical:
    """Every output-column wrapper accepts ``as_`` as a keyword argument."""

    def test_calculate_accepts_as_kwarg(self):
        # as_ is the *first positional* arg for transform_calculate, so also
        # accepted by keyword.
        result = transform_calculate(as_="ratio", expr="datum.x / datum.y")
        assert result["as_field"] == "ratio"

    def test_bin_accepts_as_kwarg(self):
        result = transform_bin("price", as_="price_bin")
        assert result["as_"] == "price_bin"

    def test_fold_accepts_as_kwarg(self):
        result = transform_fold(["a", "b"], as_=("var", "val"))
        assert result["as_"] == ["var", "val"]

    def test_density_accepts_as_kwarg(self):
        result = transform_density("x", as_=("grid", "kde"))
        assert result["as_"] == ["grid", "kde"]

    def test_regression_accepts_as_kwarg(self):
        result = transform_regression("x", "y", as_=("pred_x", "pred_y"))
        assert result["as_"] == ["pred_x", "pred_y"]

    def test_loess_accepts_as_kwarg(self):
        result = transform_loess("x", "y", as_=("smooth_x", "smooth_y"))
        assert result["as_"] == ["smooth_x", "smooth_y"]

    def test_flatten_accepts_as_kwarg(self):
        result = transform_flatten(["tags"], as_=["tag"])
        assert result["as_"] == ["tag"]

    def test_stack_accepts_as_kwarg(self):
        result = transform_stack("sales", groupby=["x"], as_=("start", "end"))
        assert result["as_"] == ["start", "end"]

    def test_timeunit_accepts_as_kwarg(self):
        result = transform_timeunit("date", "month", as_="month_date")
        assert result["as_"] == "month_date"


# ---------------------------------------------------------------------------
# 2. Wire output byte-identity: emitted keys are unchanged
# ---------------------------------------------------------------------------


class TestWireKeyIdentity:
    """Emitted wire dict keys match the pre-T3.2 contract exactly."""

    def test_calculate_wire_key_is_as_field(self):
        # INTENTIONAL: transform_calculate emits "as_field", not "as_".
        # This is a wire-key inconsistency with the other wrappers (which emit
        # "as_") but is NOT changed here — see report for details.
        result = transform_calculate("col", "datum.x * 2")
        assert "as_field" in result
        assert result["as_field"] == "col"
        assert "as_" not in result

    def test_bin_wire_key_is_as_(self):
        result = transform_bin("price", as_="price_bin")
        assert "as_" in result
        assert result["as_"] == "price_bin"

    def test_fold_wire_key_is_as_(self):
        result = transform_fold(["a", "b"])
        assert "as_" in result
        assert result["as_"] == ["key", "value"]

    def test_density_wire_key_is_as_(self):
        result = transform_density("x")
        assert "as_" in result
        assert result["as_"] == ["value", "density"]

    def test_regression_wire_key_is_as_(self):
        result = transform_regression("x", "y")
        assert "as_" in result
        assert result["as_"] == ["x", "y"]

    def test_loess_wire_key_is_as_(self):
        result = transform_loess("x", "y")
        assert "as_" in result
        assert result["as_"] == ["x", "y"]

    def test_stack_wire_key_is_as_(self):
        result = transform_stack("val", groupby=["x"])
        assert "as_" in result
        assert result["as_"] == ["y0", "y1"]

    def test_timeunit_wire_key_is_as_(self):
        result = transform_timeunit("date", "year", as_="year_col")
        assert "as_" in result
        assert result["as_"] == "year_col"

    def test_timeunit_omits_as_when_not_provided(self):
        result = transform_timeunit("date", "year")
        assert "as_" not in result

    def test_bin_omits_as_when_not_provided(self):
        result = transform_bin("price")
        assert "as_" not in result

    def test_flatten_omits_as_when_not_provided(self):
        result = transform_flatten(["tags"])
        assert "as_" not in result


# ---------------------------------------------------------------------------
# 3. Validation parity — invalid as_ shapes raise ValueError consistently
# ---------------------------------------------------------------------------


class TestValidationParityStr:
    """Single-column as_ wrappers reject invalid values with ValueError."""

    @pytest.mark.parametrize("bad", ["", 42, None, [], ("a", "b")])
    def test_calculate_rejects_bad_as(self, bad):
        with pytest.raises(ValueError, match="as_"):
            transform_calculate(bad, "datum.x")

    @pytest.mark.parametrize("bad", ["", 42, [], ("a", "b")])
    def test_bin_rejects_bad_as(self, bad):
        with pytest.raises(ValueError, match="as_"):
            transform_bin("price", as_=bad)

    @pytest.mark.parametrize("bad", ["", 42, []])
    def test_timeunit_rejects_bad_as(self, bad):
        with pytest.raises(ValueError, match="as_"):
            transform_timeunit("date", "month", as_=bad)


class TestValidationParityPair:
    """Two-column as_ wrappers reject invalid values with ValueError."""

    @pytest.mark.parametrize(
        "bad",
        [
            ("a",),  # only one element
            ("a", "b", "c"),  # three elements
            ("a", ""),  # empty second element
            ("", "b"),  # empty first element
            "ab",  # a string, not a pair
            42,  # not a sequence
        ],
    )
    def test_fold_rejects_bad_as(self, bad):
        with pytest.raises(ValueError, match="as_"):
            transform_fold(["x", "y"], as_=bad)

    @pytest.mark.parametrize(
        "bad",
        [
            ("a",),
            ("a", "b", "c"),
            ("a", ""),
            42,
        ],
    )
    def test_density_rejects_bad_as(self, bad):
        with pytest.raises(ValueError, match="as_"):
            transform_density("x", as_=bad)

    @pytest.mark.parametrize(
        "bad",
        [
            ("a",),
            ("a", "b", "c"),
            ("", "b"),
            42,
        ],
    )
    def test_regression_rejects_bad_as(self, bad):
        with pytest.raises(ValueError, match="as_"):
            transform_regression("x", "y", as_=bad)

    @pytest.mark.parametrize(
        "bad",
        [
            ("a",),
            ("a", "b", "c"),
            ("a", ""),
            42,
        ],
    )
    def test_loess_rejects_bad_as(self, bad):
        with pytest.raises(ValueError, match="as_"):
            transform_loess("x", "y", as_=bad)

    @pytest.mark.parametrize(
        "bad",
        [
            ("a",),
            ("a", "b", "c"),
            ("", "b"),
            42,
        ],
    )
    def test_stack_rejects_bad_as(self, bad):
        with pytest.raises(ValueError, match="as_"):
            transform_stack("val", groupby=["x"], as_=bad)


class TestValidationParitySeq:
    """Variable-length as_ wrappers reject invalid values with ValueError."""

    @pytest.mark.parametrize(
        "bad",
        [
            [],  # empty sequence
            [""],  # empty-string element
            ["a", ""],  # mixed valid/empty
            42,  # not a sequence
        ],
    )
    def test_flatten_rejects_bad_as(self, bad):
        with pytest.raises(ValueError, match="as_"):
            transform_flatten(["tags"], as_=bad)


# ---------------------------------------------------------------------------
# 4. Valid as_ shapes are accepted (guard against over-rejection)
# ---------------------------------------------------------------------------


class TestValidAsShapesAccepted:
    """Confirm that currently-valid as_ values still pass validation."""

    def test_calculate_valid(self):
        assert transform_calculate("my_col", "datum.x")["as_field"] == "my_col"

    def test_bin_valid_str(self):
        assert transform_bin("price", as_="p_bin")["as_"] == "p_bin"

    def test_fold_valid_tuple(self):
        assert transform_fold(["a", "b"], as_=("key", "value"))["as_"] == ["key", "value"]

    def test_fold_valid_list(self):
        # Lists are also acceptable (the type hint says tuple but a list is valid)
        assert transform_fold(["a", "b"], as_=["variable", "measure"])["as_"] == [
            "variable",
            "measure",
        ]

    def test_density_defaults_accepted(self):
        result = transform_density("x")
        assert result["as_"] == ["value", "density"]

    def test_regression_defaults_accepted(self):
        result = transform_regression("x", "y")
        assert result["as_"] == ["x", "y"]

    def test_loess_defaults_accepted(self):
        result = transform_loess("x", "y")
        assert result["as_"] == ["x", "y"]

    def test_stack_defaults_accepted(self):
        result = transform_stack("val", groupby=["x"])
        assert result["as_"] == ["y0", "y1"]

    def test_flatten_list_accepted(self):
        result = transform_flatten(["a", "b"], as_=["col_a", "col_b"])
        assert result["as_"] == ["col_a", "col_b"]

    def test_timeunit_str_accepted(self):
        result = transform_timeunit("ts", "year", as_="year_ts")
        assert result["as_"] == "year_ts"


# ---------------------------------------------------------------------------
# 5. Dict-list wrappers: inner "as" key is Vega wire form (not drift)
# ---------------------------------------------------------------------------


class TestInnerDictAsKeyIsVegaWireForm:
    """transform_aggregate, transform_join_aggregate, transform_window pass
    user inner-dicts verbatim; the ``"as"`` key inside is the Vega wire form."""

    def test_aggregate_inner_dict_as_key_passes_through(self):
        from ferrum.transforms import transform_aggregate

        agg = {"field": "price", "fn": "mean", "as": "avg_price"}
        result = transform_aggregate(agg, groupby=["cat"])
        # The inner dict is passed through unchanged — "as" (no underscore)
        assert result["aggregates"][0]["as"] == "avg_price"

    def test_join_aggregate_inner_dict_as_key_passes_through(self):
        from ferrum.transforms import transform_join_aggregate

        agg = {"field": "x", "fn": "sum", "as": "total"}
        result = transform_join_aggregate(agg)
        assert result["aggregates"][0]["as"] == "total"

    def test_window_inner_dict_as_key_passes_through(self):
        from ferrum.transforms import transform_window

        op = {"op": "row_number", "as": "rank"}
        result = transform_window(op, sort=["score"])
        assert result["ops"][0]["as"] == "rank"


# ---------------------------------------------------------------------------
# 6. One-shot iterable regression (T3.2 fix — double-consume bug)
# ---------------------------------------------------------------------------


class TestOneShotIterableNotConsumedTwice:
    """Generators and other one-shot iterables passed as ``as_`` must not be
    silently exhausted during validation, leaving an empty list in the wire
    output.  These tests would FAIL before the fix (emitting ``as_=[]``)."""

    def test_flatten_seq_generator_not_double_consumed(self):
        """transform_flatten with a generator as_ emits the correct list."""
        result = transform_flatten(["tags"], as_=(c for c in ["a", "b"]))
        assert result["as_"] == ["a", "b"], (
            f"Expected ['a', 'b'], got {result['as_']!r}. "
            "Generator was likely consumed twice (double-consume bug)."
        )

    def test_fold_pair_generator_not_double_consumed(self):
        """transform_fold with a two-element generator as_ emits the correct list."""
        result = transform_fold(["x", "y"], as_=(c for c in ["key", "value"]))
        assert result["as_"] == ["key", "value"], (
            f"Expected ['key', 'value'], got {result['as_']!r}. "
            "Generator was likely consumed twice (double-consume bug)."
        )
