"""End-to-end coverage for the expression-grammar additions (F-L05-06,
batch-D task 3): the ``,`` token, array literals, ``indexof(...)`` calls, and
``transform_filter``'s dict-shorthand branches emitting expression-grammar
literals instead of Python ``repr()`` output.

The Rust half (``crates/ferrum-core/src/transform/expr.rs``) has its own unit
suite for the grammar itself (tokenizer/parser/eval, arity errors, unknown
functions, the previously-valid-corpus regression sweep). This file is the
Python half: it drives the same grammar through the real production entry
point (``Chart.to_svg``), pinning that ``fm.transform_filter({...})``'s
documented dict-list shortcut — and the scalar/comparison-dict shortcuts,
which share the same literal-serialization code path — produce predicates
the grammar can actually parse, for every literal type the grammar supports
(strings, numbers, bools, ``None``/``null``).

Round 2 additions (batch-D task 3, python-quality-reviewer required fixes
1/2/4): the ``_expr_literal`` numeric-acceptance boundary (numpy scalars
normalize to plain literals; ``Decimal``/``datetime`` still refuse), the
non-finite-float refusal, and e2e round-trips for escaping and non-ASCII
values (the latter now works — see the Rust round-2 UTF-8 tokenizer fix in
``expr.rs``).
"""

from __future__ import annotations

import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

import ferrum as fm
from ferrum.transforms import _expr_literal, transform_filter


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "country": ["US", "DE", "JP", "FR", "CA"],
            "active": [True, False, True, True, False],
            "tier": ["gold", None, "silver", "gold", None],
        }
    )


def _n_circles(chart: "fm.Chart") -> int:
    return chart.to_svg().count("<circle")


def _filtered_chart(predicate) -> "fm.Chart":
    df = _df()
    return (
        fm.Chart(df).mark_point().encode(x="id", y="id").transform(fm.transform_filter(predicate))
    )


class TestDictListFilterLiterals:
    """The guide's documented shortcut: ``transform_filter({field: [...]})``."""

    def test_guide_example_string_membership(self):
        # fm.transform_filter({"country": ["US", "DE", "JP"]}) — the guide's
        # exact documented shortcut.
        chart = _filtered_chart({"country": ["US", "DE", "JP"]})
        assert _n_circles(chart) == 3

    def test_numeric_membership(self):
        chart = _filtered_chart({"id": [1, 3, 5]})
        assert _n_circles(chart) == 3

    def test_bool_membership_true_only(self):
        chart = _filtered_chart({"active": [True]})
        assert _n_circles(chart) == 3

    def test_bool_membership_false_only(self):
        chart = _filtered_chart({"active": [False]})
        assert _n_circles(chart) == 2

    def test_none_membership(self):
        # None list elements must serialize as the grammar's `null` literal,
        # not Python's `None` (an unrecognized bare identifier to the parser).
        chart = _filtered_chart({"tier": [None, "gold"]})
        assert _n_circles(chart) == 4  # 2 gold + 2 null tier rows

    def test_predicate_text_uses_grammar_literals(self):
        result = transform_filter({"country": ["US", "DE"], "active": [True]})
        predicate = result["predicate"]
        assert '"US"' in predicate
        assert '"DE"' in predicate
        assert "true" in predicate
        # Python's capitalized repr forms must never leak into the predicate.
        assert "True" not in predicate
        assert "'US'" not in predicate


class TestRawIndexofExpression:
    """The other documented form: a raw ``indexof(...)`` expression string."""

    def test_raw_indexof_membership(self):
        chart = _filtered_chart("indexof([1, 3, 5], datum.id) >= 0")
        assert _n_circles(chart) == 3

    def test_raw_indexof_string_membership(self):
        chart = _filtered_chart('indexof(["US", "DE"], datum.country) >= 0')
        assert _n_circles(chart) == 2

    def test_raw_indexof_not_found_sentinel(self):
        chart = _filtered_chart("indexof([99, 100], datum.id) >= 0")
        assert _n_circles(chart) == 0


class TestPreviouslyValidFormsRegression:
    """A previously-valid-forms sweep at the Python level: scalar filter,
    comparison dict, and plain expression strings must all still work
    end to end after the literal-serialization fix.
    """

    def test_scalar_string_filter(self):
        chart = _filtered_chart({"country": "US"})
        assert _n_circles(chart) == 1

    def test_scalar_numeric_filter(self):
        chart = _filtered_chart({"id": 3})
        assert _n_circles(chart) == 1

    def test_scalar_bool_filter(self):
        # The scalar branch shares _expr_literal with the dict-list branch;
        # it had the identical Python-repr bug for bool/None values even
        # though it wasn't the documented F-L05-06 shortcut.
        chart = _filtered_chart({"active": True})
        assert _n_circles(chart) == 3

    def test_comparison_dict_filter(self):
        chart = _filtered_chart({"id": {">=": 3}})
        assert _n_circles(chart) == 3

    def test_comparison_dict_bool_filter(self):
        chart = _filtered_chart({"active": {"==": False}})
        assert _n_circles(chart) == 2

    def test_plain_expression_string(self):
        chart = _filtered_chart("datum.id > 2 && datum.id < 5")
        assert _n_circles(chart) == 2


class TestExprLiteralHelper:
    """Unit-level coverage of the literal serializer itself."""

    def test_string_literal_is_quoted(self):
        assert _expr_literal("US") == '"US"'

    def test_string_literal_escapes_embedded_quotes(self):
        assert _expr_literal('say "hi"') == '"say \\"hi\\""'

    def test_bool_literals_are_lowercase(self):
        assert _expr_literal(True) == "true"
        assert _expr_literal(False) == "false"

    def test_none_is_null(self):
        assert _expr_literal(None) == "null"

    def test_numbers_are_plain(self):
        assert _expr_literal(5) == "5"
        assert _expr_literal(5.5) == "5.5"

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            _expr_literal(object())

    def test_numpy_float64_normalizes_to_plain_literal(self):
        # numpy >= 2's repr(np.float64(2.0)) is "np.float64(2.0)", which the
        # grammar's tokenizer cannot parse. df[col].mean()/.max()/.quantile()
        # returning a numpy scalar is the routine way a user gets a
        # comparison-dict threshold, so this is the live path, not a corner.
        assert _expr_literal(np.float64(2.0)) == "2.0"
        assert _expr_literal(np.float32(1.5)) == "1.5"

    def test_numpy_int64_now_accepted(self):
        # Resolves the asymmetry: np.int64 is not an `int` subclass (unlike
        # np.float64, which *is* a `float` subclass), so it used to hit the
        # TypeError refusal branch while its float twin sailed through with
        # a hostile repr. Both are numeric and both must now normalize.
        assert _expr_literal(np.int64(3)) == "3"

    def test_large_int_emitted_literal_is_exact(self):
        # The encoder emits the exact digit string (int and float normalize
        # via separate paths: operator.index vs. float() — this separation
        # ensures the encoder never routes a large integer through float()
        # and loses precision at serialization time). However, the grammar
        # evaluates all numbers as f64, so a large-int filter can still match
        # its f64-equivalent neighbor end-to-end — that is a grammar-side
        # evaluation caveat, not an encoder defect.
        big = 2**60 + 1
        assert _expr_literal(np.int64(big)) == repr(big)

    def test_numpy_bool_normalizes_like_python_bool(self):
        # np.bool_ is not a bool subclass, so it must be accepted alongside
        # Python bool. df[col].iloc[0] (for a boolean column) and df[col]
        # .unique() (for categorical/boolean columns) return numpy scalars,
        # which is how a user gets a boolean to feed to the dict-shorthand
        # filter. Both True and False cases must work.
        assert _expr_literal(np.True_) == "true"
        assert _expr_literal(np.False_) == "false"

    def test_non_finite_float_raises_naming_field(self):
        with pytest.raises(ValueError, match="threshold") as excinfo:
            _expr_literal(float("inf"), field="threshold")
        assert "inf" in str(excinfo.value)

    def test_nan_raises(self):
        with pytest.raises(ValueError):
            _expr_literal(float("nan"))

    def test_non_finite_numpy_float_raises(self):
        with pytest.raises(ValueError):
            _expr_literal(np.float64("-inf"))

    def test_decimal_boundary_case_still_raises(self):
        # decimal.Decimal looks scalar but is deliberately not registered
        # under numbers.Real by the stdlib itself (mixing Decimal and float
        # arithmetic is unsafe), so it must still hit the TypeError branch
        # rather than being silently coerced through float().
        with pytest.raises(TypeError):
            _expr_literal(Decimal("2.5"))

    def test_datetime_boundary_case_still_raises(self):
        with pytest.raises(TypeError):
            _expr_literal(datetime.datetime(2026, 1, 1))


class TestEscapingRoundTrip:
    """E2E round-trips through ``to_svg`` for escaped string content —
    py-quality Required Fix 4. A prior version of this suite only pinned
    the *encoder's* output string (``test_string_literal_escapes_embedded_
    quotes`` above); that leaves the grammar's decode side unverified, so a
    change that kept the unit test green (e.g. switching to unescaped
    single-quote wrapping) could still silently break real filters. These
    tests filter a DataFrame column containing the hostile value and assert
    the row count, so a decode mismatch shows up as a wrong filtered row.
    """

    def _df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "id": [1, 2, 3],
                "text": ['say "hi"', "a\\b", "plain"],
            }
        )

    def _n_circles(self, predicate) -> int:
        df = self._df()
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(x="id", y="id")
            .transform(fm.transform_filter(predicate))
        )
        return chart.to_svg().count("<circle")

    def test_embedded_double_quote_round_trips(self):
        assert self._n_circles({"text": ['say "hi"']}) == 1

    def test_embedded_backslash_round_trips(self):
        assert self._n_circles({"text": ["a\\b"]}) == 1

    def test_non_ascii_value_round_trips(self):
        # Previously silently matched 0 rows (byte-wise Latin-1 tokenizer
        # decode); now fixed Rust-side (round-2 UTF-8-aware string
        # tokenizer), so this filters exactly the one matching row.
        df = pd.DataFrame({"id": [1, 2], "city": ["Zürich", "Paris"]})
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(x="id", y="id")
            .transform(fm.transform_filter({"city": ["Zürich"]}))
        )
        assert chart.to_svg().count("<circle") == 1


class TestNumericNormalizationEndToEnd:
    """Numeric-subclass and non-finite-float coverage through the real
    production path — py-quality Required Fixes 1 and 2.
    """

    def test_numpy_aggregate_threshold_through_to_svg(self):
        # df[col].mean() is the canonical way a user gets a threshold to
        # feed a comparison-dict filter; it returns a numpy scalar, not a
        # plain Python float.
        df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
        threshold = df["id"].mean()
        assert isinstance(threshold, np.floating)
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(x="id", y="id")
            .transform(fm.transform_filter({"id": {">=": threshold}}))
        )
        assert chart.to_svg().count("<circle") == 3  # ids 3, 4, 5

    def test_non_finite_float_refused_at_transform_filter(self):
        # The refusal happens at transform_filter construction time (the
        # dict branch calls _expr_literal eagerly), not 11 characters into
        # a grammar parse error surfaced later at render time.
        with pytest.raises(ValueError, match="id"):
            transform_filter({"id": {">=": float("inf")}})
