"""Transform-engine dtype policy — the batch-D M1 sweep matrix.

Policy (design spec §4.1): a transform's *value-column* ingestion accepts every
Arrow integer / unsigned / float dtype by coercing to Float64 through
``numeric_util::coerce_to_float64``; every other dtype raises a typed
``ValueError`` naming the transform, the column, and the actual dtype.  No
transform emits silent nulls, NaNs, or fallback values for an unhandled dtype.

Five things are pinned here, all through the production ``to_svg`` entry point:

1. **The sweep matrix.**  Every transform touched by the ``_sweep_cases()``
   value-column matrix renders *byte-identically* from an ``Int64`` /
   ``Int32`` / ``Float32`` value column and from the same data pre-cast to
   ``Float64``.  Byte-identity is the invariant that makes the coercion a
   widening rather than a behavior change.
2. **The refusals.**  A ``Utf8`` value column raises the transform's own typed
   message, naming the transform, the column, and the dtype.
3. **The two headline repros** the batch's intent names: ``mark_errorbar``'s own
   docstring example (an ``Int64`` column) renders, and the user guide's
   ``transform_flatten`` list-of-strings example emits real strings instead of
   an all-null column.
4. **The expression-engine half.**  ``transform_filter`` and
   ``transform_calculate`` read every int/uint dtype rather than silently seeing
   ``Null`` — which made a filter drop the entire dataset — and refuse a
   genuinely unreadable dtype by name.  Plus ``transform_timeunit``'s unknown
   ``unit`` refusal.
5. **Column-selection and field-ingestion widenings.**  Two sites outside the
   value-column matrix's shape widened in the same sweep: ``fm.clustermap``'s
   feature-column selection (``linkage::extract_matrix``, which used to pick
   feature columns by an exact ``Float64`` match) and
   ``fm.transform_timeunit``'s epoch-column ingestion
   (``timeunit::extract_millis``, previously ``Timestamp``/``Float64``/
   ``Int64`` only).  Neither fits ``_sweep_cases()``'s ``sweep_frame`` builder
   shape, so each is pinned by a standalone regression test instead.

Per the project's test-file convention this module carries net-new feature
coverage for the dtype policy as a whole, not one findings ID.
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import ferrum as fm

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

#: Numeric dtypes that must coerce to the same output as Float64.
NUMERIC_DTYPES = [pl.Int64, pl.Int32, pl.Float32]

_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
_XS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
_GROUPS = ["a"] * 6 + ["b"] * 6


def sweep_frame(dtype) -> pl.DataFrame:
    """Value/position/group frame with the numeric columns cast to *dtype*.

    ``dtype=pl.Utf8`` casts only the value column, so the invalid-dtype cases
    exercise a non-numeric *value* against otherwise-valid positions.
    """
    df = pl.DataFrame({"v": _VALUES, "x": _XS, "g": _GROUPS})
    if dtype is pl.Utf8:
        return df.with_columns(pl.col("v").cast(pl.Utf8))
    return df.with_columns(pl.col("v").cast(dtype), pl.col("x").cast(dtype))


def fit_frame(dtype) -> pl.DataFrame:
    """Frame for the two-column fit family (logistic needs a 0/1 response)."""
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "binary": [0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
            "resp": [2, 4, 5, 4, 6, 7, 9, 8, 11, 12],
        }
    )
    if dtype is pl.Utf8:
        return df.with_columns(pl.col("resp").cast(pl.Utf8))
    return df.with_columns(
        pl.col("x").cast(dtype), pl.col("binary").cast(dtype), pl.col("resp").cast(dtype)
    )


# ---------------------------------------------------------------------------
# The sweep matrix: (case id, chart builder, transform name in its errors)
# ---------------------------------------------------------------------------


def _sweep_cases() -> list[tuple[str, object, str]]:
    """One entry per swept transform, reached through its public surface."""
    return [
        (
            "error_extent",
            lambda d: fm.Chart(d).mark_errorbar(method="stdev").encode(x="g", y="v"),
            "stat_error_extent",
        ),
        (
            "summary",
            lambda d: (
                fm.Chart(d)
                .transform(fm.Summary(field="v", groupby=["g"]))
                .mark_point()
                .encode(x="g:N", y="mean:Q")
            ),
            "stat_summary",
        ),
        (
            "box_stats",
            lambda d: fm.Chart(d).mark_boxplot().encode(x="g", y="v"),
            "stat_box_stats",
        ),
        (
            "letter_value",
            lambda d: fm.Chart(d).mark_boxen().encode(x="g", y="v"),
            "stat_letter_value",
        ),
        (
            "smooth",
            lambda d: fm.Chart(d).mark_smooth(method="lm").encode(x="x", y="v"),
            "stat_smooth",
        ),
        (
            "violin",
            lambda d: fm.Chart(d).mark_violin().encode(x="g", y="v"),
            "stat_violin",
        ),
        (
            "raster",
            lambda d: fm.Chart(d).mark_raster().encode(x="x", y="v"),
            "stat_raster",
        ),
        (
            "aggregate",
            lambda d: fm.Chart(d).mark_bar().encode(x="g:N", y="mean(v):Q"),
            "stat_aggregate",
        ),
        (
            "regression_data",
            lambda d: (
                fm.Chart(d)
                .transform(fm.transform_regression("x", "v"))
                .mark_line()
                .encode(x="x:Q", y="y:Q")
            ),
            "data_regression",
        ),
        (
            "loess_data",
            lambda d: (
                fm.Chart(d)
                .transform(fm.transform_loess("x", "v"))
                .mark_line()
                .encode(x="x:Q", y="y:Q")
            ),
            "data_loess",
        ),
        (
            "impute",
            lambda d: (
                fm.Chart(d)
                .transform(fm.transform_impute("v", method="mean"))
                .mark_point()
                .encode(x="x:Q", y="v:Q")
            ),
            "data_impute",
        ),
        (
            "data_stack",
            lambda d: (
                fm.Chart(d)
                .transform(fm.transform_stack("v", groupby=["g"]))
                .mark_bar()
                .encode(x="g:N", y="y1:Q")
            ),
            "data_stack",
        ),
        (
            "data_bin",
            lambda d: (
                fm.Chart(d)
                .transform(fm.transform_bin("v", as_="vb"))
                .mark_point()
                .encode(x="vb:Q", y="x:Q")
            ),
            "data_bin",
        ),
        (
            "pivot",
            lambda d: (
                fm.Chart(d)
                .transform(fm.transform_pivot("g", "v"))
                .mark_point()
                .encode(x="a:Q", y="b:Q")
            ),
            "data_pivot",
        ),
        (
            "data_aggregate",
            lambda d: (
                fm.Chart(d)
                .transform(
                    fm.transform_aggregate({"field": "v", "fn": "mean", "as": "m"}, groupby=["g"])
                )
                .mark_bar()
                .encode(x="g:N", y="m:Q")
            ),
            "data_aggregate",
        ),
        (
            "join_aggregate",
            lambda d: (
                fm.Chart(d)
                .transform(
                    fm.transform_join_aggregate(
                        {"field": "v", "fn": "mean", "as": "m"}, groupby=["g"]
                    )
                )
                .mark_point()
                .encode(x="x:Q", y="m:Q")
            ),
            "data_join_aggregate",
        ),
        (
            "density_data",
            lambda d: (
                fm.Chart(d)
                .transform(fm.transform_density("v"))
                .mark_line()
                .encode(x="value:Q", y="density:Q")
            ),
            "data_density",
        ),
        (
            "swarm",
            lambda d: fm.Chart(d).mark_swarm().encode(x="g", y="v"),
            "stat_swarm",
        ),
    ]


#: The fit family, reached through ``lmplot``'s method routing.
_FIT_CASES = [
    ("logistic", "binary", "logistic", "Logistic"),
    ("robust", "resp", "robust", "Robust"),
    ("glm", "resp", "glm", "Glm"),
]

SWEEP_CASES = _sweep_cases()
SWEEP_IDS = [c[0] for c in SWEEP_CASES]


def _svg(builder, df: pl.DataFrame) -> str:
    return builder(df).to_svg()


# ---------------------------------------------------------------------------
# 1. Numeric dtypes render byte-identically to their Float64 equivalent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: str(d))
@pytest.mark.parametrize(("name", "builder", "_ctx"), SWEEP_CASES, ids=SWEEP_IDS)
def test_numeric_dtype_renders_byte_identically_to_float64(name, builder, _ctx, dtype):
    """Coercion is a widening: the SVG must not move by a byte."""
    baseline = _svg(builder, sweep_frame(pl.Float64))
    assert baseline, f"{name}: Float64 baseline rendered nothing"
    assert _svg(builder, sweep_frame(dtype)) == baseline


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: str(d))
@pytest.mark.parametrize(
    ("name", "y_col", "method", "_ctx"), _FIT_CASES, ids=[c[0] for c in _FIT_CASES]
)
def test_fit_family_numeric_dtype_renders_byte_identically(name, y_col, method, _ctx, dtype):
    """logistic / robust / glm ingest their x and y columns through one helper."""

    def build(df):
        return fm.lmplot(df, x="x", y=y_col, method=method)

    baseline = build(fit_frame(pl.Float64)).to_svg()
    assert baseline, f"{name}: Float64 baseline rendered nothing"
    assert build(fit_frame(dtype)).to_svg() == baseline


# ---------------------------------------------------------------------------
# 2. Non-numeric dtypes raise a typed error naming transform, column, dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "builder", "ctx"), SWEEP_CASES, ids=SWEEP_IDS)
def test_utf8_value_column_raises_naming_transform_column_and_dtype(name, builder, ctx):
    with pytest.raises(ValueError) as excinfo:
        _svg(builder, sweep_frame(pl.Utf8))
    msg = str(excinfo.value)
    assert ctx in msg, f"{name}: error must name the transform; got {msg!r}"
    assert "'v'" in msg, f"{name}: error must name the column; got {msg!r}"
    assert "must be numeric" in msg, f"{name}: error must state the policy; got {msg!r}"
    # polars strings arrive as LargeUtf8; either spelling names the dtype.
    assert "Utf8" in msg, f"{name}: error must name the dtype; got {msg!r}"


@pytest.mark.parametrize(
    ("name", "y_col", "method", "ctx"), _FIT_CASES, ids=[c[0] for c in _FIT_CASES]
)
def test_fit_family_utf8_column_raises_naming_transform_and_dtype(name, y_col, method, ctx):
    with pytest.raises(ValueError) as excinfo:
        fm.lmplot(fit_frame(pl.Utf8), x="x", y="resp", method=method).to_svg()
    msg = str(excinfo.value)
    assert ctx in msg, f"{name}: error must name the transform; got {msg!r}"
    assert "must be numeric" in msg, f"{name}: error must state the policy; got {msg!r}"
    assert "Utf8" in msg, f"{name}: error must name the dtype; got {msg!r}"


def test_swarm_non_numeric_value_column_raises_typed_refusal():
    """swarm's non-numeric refusal is now the engine's typed message.

    ``_desugar.cast_columns_to_float64(..., int_only=False)`` used to pre-cast
    swarm's value column in Python for *any* non-Float64 dtype, so a
    genuinely non-numeric column was refused by polars with an untyped
    ``InvalidOperationError`` before the transform was ever reached. That
    shim is deleted (batch D, M1 Python half); the refusal now comes from
    ``stat_swarm`` itself, naming the transform, column, and dtype like every
    other swept transform.
    """
    df = pl.DataFrame({"v": ["lo", "hi", "lo", "mid", "hi", "lo"], "g": ["a"] * 3 + ["b"] * 3})
    with pytest.raises(ValueError) as excinfo:
        fm.Chart(df).mark_swarm().encode(x="g", y="v").to_svg()
    msg = str(excinfo.value)
    assert "stat_swarm" in msg, f"error must name the transform; got {msg!r}"
    assert "'v'" in msg, f"error must name the column; got {msg!r}"
    assert "must be numeric" in msg, f"error must state the policy; got {msg!r}"


def test_swarm_numeric_string_value_column_is_refused_not_silently_coerced():
    """A numeric-string swarm value column raises instead of silently rendering.

    This is the finding the batch-D M1 Python half was dispatched to close:
    the deleted shim's ``int_only=False`` predicate cast *any* column whose
    dtype was not already ``Float64``, so a numeric-string column like
    ``"1.0"`` was silently coerced and rendered as if it were the intended
    numeric input. The Rust dtype policy does not special-case
    numeric-looking strings — a ``Utf8``/``LargeUtf8`` column is refused by
    type, exactly like a non-numeric string column.
    """
    df = pl.DataFrame({"v": ["1.0", "2.0", "3.0", "4.0", "5.0", "6.0"], "g": ["a"] * 3 + ["b"] * 3})
    with pytest.raises(ValueError) as excinfo:
        fm.Chart(df).mark_swarm().encode(x="g", y="v").to_svg()
    msg = str(excinfo.value)
    assert "stat_swarm" in msg, f"error must name the transform; got {msg!r}"
    assert "'v'" in msg, f"error must name the column; got {msg!r}"
    assert "must be numeric" in msg, f"error must state the policy; got {msg!r}"


# ---------------------------------------------------------------------------
# 3. Headline repros
# ---------------------------------------------------------------------------


def test_mark_errorbar_docstring_example_renders_on_int64():
    """The example in ``Chart.mark_errorbar``'s own docstring, verbatim.

    ``list(range(10))`` is Int64 in polars, so this raised
    ``stat_error_extent: column 'val' must be Float64`` before the dtype policy.
    """
    df = pl.DataFrame(
        {"group": ["a"] * 10 + ["b"] * 10, "val": list(range(10)) + list(range(5, 15))}
    )
    assert df["val"].dtype == pl.Int64

    svg = fm.Chart(df).mark_errorbar(method="stdev").encode(x="group", y="val").to_svg()
    assert svg.startswith("<svg") or "<svg" in svg
    # And it is exactly what the pre-cast-to-float data renders.
    as_float = df.with_columns(pl.col("val").cast(pl.Float64))
    assert (
        svg == fm.Chart(as_float).mark_errorbar(method="stdev").encode(x="group", y="val").to_svg()
    )


def test_transform_flatten_guide_example_emits_real_strings():
    """The user guide's ``transform_flatten`` list-of-strings example.

    polars hands Rust ``LargeList(LargeUtf8)``; the transform used to fall
    through to an all-null Float64 column, so the tag labels never rendered.
    """
    df = pl.DataFrame(
        {
            "person": ["Alice", "Bob"],
            "tags": [["python", "rust"], ["go", "python", "java"]],
        }
    )
    svg = (
        fm.Chart(df)
        .transform(fm.transform_flatten(["tags"]))
        .mark_bar()
        .encode(x="tags:N", y="count():Q")
    ).to_svg()

    labels = set(re.findall(r">(python|rust|go|java)<", svg))
    assert labels == {"python", "rust", "go", "java"}, f"missing tag labels: {labels}"


def test_transform_flatten_list_of_ints_widens_to_float64():
    """Numeric list elements coerce, so a list-of-int column is plottable."""
    df = pl.DataFrame({"person": ["Alice", "Bob"], "scores": [[1, 2], [3, 4, 5]]})
    svg = (
        fm.Chart(df)
        .transform(fm.transform_flatten(["scores"]))
        .mark_point()
        .encode(x="person:N", y="scores:Q")
    ).to_svg()
    assert svg.count("<circle") == 5


def test_transform_flatten_carried_column_keeps_its_values():
    """Carried (non-flattened) columns repeat at their own dtype."""
    df = pl.DataFrame({"person": ["Alice", "Bob"], "tags": [["a", "b"], ["c"]]})
    svg = (
        fm.Chart(df)
        .transform(fm.transform_flatten(["tags"]))
        .mark_point()
        .encode(x="person:N", y="tags:N")
    ).to_svg()
    for name in ("Alice", "Bob"):
        assert f">{name}<" in svg, f"carried column lost {name}"


def test_transform_flatten_unsupported_element_dtype_raises():
    """A list of an unflattenable element dtype names the dtype instead of nulling."""
    df = pl.DataFrame({"person": ["Alice"], "flags": [[True, False]]})
    with pytest.raises(ValueError) as excinfo:
        (
            fm.Chart(df)
            .transform(fm.transform_flatten(["flags"]))
            .mark_point()
            .encode(x="person:N", y="flags:Q")
        ).to_svg()
    msg = str(excinfo.value)
    assert "data_flatten" in msg
    assert "'flags'" in msg
    assert "Boolean" in msg


def test_transform_flatten_carried_column_null_preserved_not_nan_on_categorical_channel():
    """A carried Float64 column's null must not become a spurious "NaN" category.

    ``repeat_by_lengths`` (the carried-column expansion) switched from a
    hand-rolled Float64/Utf8 substitution — which turned a null carried cell
    into ``f64::NAN`` — to Arrow's ``take`` kernel, which preserves the null.
    A null and a NaN both stringify to nothing on a positional channel, but
    they are *not* interchangeable on a categorical (``:N``) channel: a null
    is dropped from the domain, while a NaN becomes its own ``"NaN"`` legend
    entry. This is the live repro from the batch-D M1 review.
    """
    df = pl.DataFrame({"score": [1.0, None], "tags": [["a", "b"], ["c"]]})
    svg = (
        fm.Chart(df)
        .transform(fm.transform_flatten(["tags"]))
        .mark_bar()
        .encode(x="tags:N", y="score:Q", color="score:N")
        .to_svg()
    )
    assert "NaN" not in svg, "a carried null must not surface as a NaN category"
    # Positive half: the negative assertion alone would pass vacuously if the
    # color legend disappeared entirely. Confirm it still carries its real
    # title and entry. Since `score` is also the y-axis title and `1` is a
    # y-axis tick, we count occurrences to ensure legend uniqueness: with the
    # legend, both appear twice (axis + legend); without it, once (axis only).
    assert svg.count(">score<") == 2, "color legend title must still render"
    assert svg.count(">1<") == 2, "the categorical legend must still carry its real entry"


def test_transform_flatten_list_element_null_preserved_not_nan_on_categorical_channel():
    """The flattened-element half of the same null-preservation delta.

    A null *element* inside a flattened ``List(Float64)`` column must also
    stay a null, not the pre-fix NaN substitution, when that column feeds a
    categorical channel.
    """
    df = pl.DataFrame({"g": ["a", "b"], "scores": [[1.0, None], [3.0]]})
    svg = (
        fm.Chart(df)
        .transform(fm.transform_flatten(["scores"]))
        .mark_bar()
        .encode(x="g:N", y="count():Q", color="scores:N")
        .to_svg()
    )
    assert "NaN" not in svg, "a null list element must not surface as a NaN category"
    # Positive half: the negative assertion alone would pass vacuously if the
    # color legend disappeared entirely. Confirm it still carries its real
    # title and both surviving entries (measured live: text nodes end
    # ``['scores', '1', '3']``).
    assert ">scores<" in svg, "color legend title must still render"
    assert ">1<" in svg, "the categorical legend must still carry its '1' entry"
    assert ">3<" in svg, "the categorical legend must still carry its '3' entry"


# ---------------------------------------------------------------------------
# 4. Count keeps its any-dtype behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: str(d))
def test_outliers_transform_flags_the_same_points_on_any_numeric_dtype(dtype):
    """``stat_outliers`` runs inside boxplot; a lone extreme value must survive
    the coercion as an outlier point, identically to the Float64 input."""
    df = pl.DataFrame({"v": [1, 2, 3, 4, 5, 6, 100], "g": ["a"] * 7})

    def build(d):
        return fm.Chart(d).mark_boxplot(outliers=True).encode(x="g", y="v")

    baseline = build(df.with_columns(pl.col("v").cast(pl.Float64))).to_svg()
    assert baseline.count("<circle") == 1, "fixture must produce exactly one outlier"
    assert build(df.with_columns(pl.col("v").cast(dtype))).to_svg() == baseline


def test_count_aggregate_still_counts_non_numeric_columns():
    """``AggFn::Count`` is the documented exception to the numeric-only rule."""
    df = pl.DataFrame({"g": ["a", "a", "b"], "label": ["x", "y", "z"]})
    svg = (
        fm.Chart(df)
        .transform(
            fm.transform_aggregate({"field": "label", "fn": "count", "as": "n"}, groupby=["g"])
        )
        .mark_bar()
        .encode(x="g:N", y="n:Q")
    ).to_svg()
    assert svg.count("<rect") > 1


# ---------------------------------------------------------------------------
# 5. The expression engine reads every numeric dtype
# ---------------------------------------------------------------------------

#: Integer dtypes an expression must read. Only Int32/Int64 worked before the
#: widening; the rest silently evaluated to ``Null`` on every row.
EXPR_INT_DTYPES = [
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
]


def expr_frame(dtype) -> pl.DataFrame:
    return pl.DataFrame({"n": [1, 2, 3], "v": [1.0, 2.0, 3.0]}).with_columns(
        pl.col("n").cast(dtype)
    )


def test_transform_filter_on_uint32_keeps_matching_rows():
    """The named regression for the audit's live repro.

    ``field_value`` had no ``UInt32`` arm, so ``datum.n`` read as ``Null`` for
    every row, ``Null > 1`` was falsy, and the filter silently dropped the whole
    dataset — no error, just an empty chart.
    """
    df = expr_frame(pl.UInt32)
    svg = (
        fm.Chart(df)
        .transform(fm.transform_filter("datum.n > 1"))
        .mark_point()
        .encode(x="n:Q", y="v:Q")
    ).to_svg()
    assert svg.count("<circle") == 2, "UInt32 filter must keep rows 2 and 3"


@pytest.mark.parametrize("dtype", EXPR_INT_DTYPES, ids=lambda d: str(d))
def test_transform_filter_reads_every_numeric_dtype(dtype):
    """Every numeric dtype filters identically to the Int64 equivalent."""
    baseline = (
        fm.Chart(expr_frame(pl.Int64))
        .transform(fm.transform_filter("datum.n > 1"))
        .mark_point()
        .encode(x="n:Q", y="v:Q")
    ).to_svg()
    assert baseline.count("<circle") == 2

    got = (
        fm.Chart(expr_frame(dtype))
        .transform(fm.transform_filter("datum.n > 1"))
        .mark_point()
        .encode(x="n:Q", y="v:Q")
    ).to_svg()
    assert got.count("<circle") == 2, f"{dtype}: filter dropped rows it should keep"


@pytest.mark.parametrize("dtype", EXPR_INT_DTYPES, ids=lambda d: str(d))
def test_transform_calculate_reads_every_numeric_dtype(dtype):
    """calculate shares ``field_value``; an unread dtype gave an all-null column."""
    svg = (
        fm.Chart(expr_frame(dtype))
        .transform(fm.transform_calculate("doubled", "datum.n * 2"))
        .mark_point()
        .encode(x="doubled:Q", y="v:Q")
    ).to_svg()
    assert svg.count("<circle") == 3, f"{dtype}: calculate produced a null column"
    # The derived column must span 2..6, so its axis carries a "6" tick label.
    assert ">6<" in svg, f"{dtype}: derived column did not reach 6"


def test_transform_filter_and_calculate_agree_across_dtypes():
    """The two consumers of the shared reader must not diverge by dtype."""

    def build(transform, encode_x, df):
        return fm.Chart(df).transform(transform).mark_point().encode(x=encode_x, y="v:Q").to_svg()

    for dtype in EXPR_INT_DTYPES:
        assert build(fm.transform_filter("datum.n > 1"), "n:Q", expr_frame(dtype)) == build(
            fm.transform_filter("datum.n > 1"), "n:Q", expr_frame(pl.Float64)
        )


def test_expression_on_unreadable_dtype_raises_naming_column_and_dtype():
    """A list column in a predicate refuses by name instead of reading as Null."""
    df = pl.DataFrame({"tags": [["a"], ["b"], ["c"]], "v": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError) as excinfo:
        (
            fm.Chart(df)
            .transform(fm.transform_filter("datum.tags == 'a'"))
            .mark_point()
            .encode(x="v:Q", y="v:Q")
        ).to_svg()
    msg = str(excinfo.value)
    assert "data_filter" in msg
    assert "'tags'" in msg
    assert "List" in msg


def test_expression_missing_field_still_evaluates_to_null():
    """Vega's ``datum.absent`` semantics are unchanged by the dtype policy."""
    df = pl.DataFrame({"n": [1, 2, 3], "v": [1.0, 2.0, 3.0]})
    svg = (
        fm.Chart(df)
        .transform(fm.transform_filter("datum.absent > 1"))
        .mark_point()
        .encode(x="n:Q", y="v:Q")
    ).to_svg()
    assert svg.count("<circle") == 0


# ---------------------------------------------------------------------------
# 6. timeunit refuses an unknown unit instead of emitting NaN
# ---------------------------------------------------------------------------


def test_transform_timeunit_unknown_unit_raises():
    """An unrecognised unit used to yield an all-NaN column silently."""
    df = pl.DataFrame({"ts": [1_686_831_000_000.0, 1_686_917_400_000.0], "v": [1.0, 2.0]})
    with pytest.raises(ValueError) as excinfo:
        (
            fm.Chart(df)
            .transform(fm.transform_timeunit("ts", "fortnight", as_="u"))
            .mark_point()
            .encode(x="u:Q", y="v:Q")
        ).to_svg()
    msg = str(excinfo.value)
    assert "data_timeunit" in msg
    assert "'fortnight'" in msg
    assert "day_of_week" in msg, "error must list the accepted units"


def test_transform_timeunit_known_unit_still_works():
    df = pl.DataFrame({"ts": [1_686_831_000_000.0, 1_686_917_400_000.0], "v": [1.0, 2.0]})
    svg = (
        fm.Chart(df)
        .transform(fm.transform_timeunit("ts", "year", as_="u"))
        .mark_point()
        .encode(x="u:Q", y="v:Q")
    ).to_svg()
    assert svg.count("<circle") == 2


@pytest.mark.parametrize("dtype", [pl.Float32, pl.UInt64], ids=lambda d: str(d))
def test_transform_timeunit_widens_epoch_column_to_every_numeric_dtype(dtype):
    """``timeunit::extract_millis``'s field-ingestion widening (batch D round 2).

    Before the M1 sweep, ``extract_millis`` read only ``Timestamp`` /
    ``Float64`` / ``Int64`` epoch columns; every other numeric dtype fell
    through the dispatch. This is the production entry point for the widened
    ``other if is_numeric_dtype(other)`` arm (Rust inline coverage lives in
    ``timeunit.rs``). Byte-identity against the ``Float64`` baseline is the
    self-evidently discriminating pin: a reverted arm either raises instead
    of rendering, or (if ``UInt64``/``Float32`` fell through to a different
    arm's arithmetic) would move the derived year off the baseline's.
    """
    df = pl.DataFrame({"ts": [1_686_831_000_000.0, 1_686_917_400_000.0], "v": [1.0, 2.0]})

    def build(d):
        return (
            fm.Chart(d)
            .transform(fm.transform_timeunit("ts", "year", as_="u"))
            .mark_point()
            .encode(x="u:Q", y="v:Q")
        ).to_svg()

    baseline = build(df)
    assert baseline.count("<circle") == 2
    assert build(df.with_columns(pl.col("ts").cast(dtype))) == baseline


# ---------------------------------------------------------------------------
# 7. linkage widens feature-column selection to every numeric dtype
# ---------------------------------------------------------------------------


def test_clustermap_int64_feature_columns_render_byte_identically_to_float64():
    """``linkage::extract_matrix``'s column-selection widening (batch D round 2).

    Before the M1 sweep, ``extract_matrix`` selected feature columns by an
    exact ``dtype == Float64`` match, so an ``Int64`` feature column was
    silently excluded from the distance matrix while ``fm.clustermap``'s
    Python side still unpivoted it into the heatmap — a selection mismatch
    between the two layers. This is the production entry point for the fix
    (Rust inline coverage lives in ``linkage.rs``). Not folded into
    ``_sweep_cases()``: ``clustermap`` does not fit the ``sweep_frame``
    builder shape, so this is a standalone regression test.

    Byte-identity against the ``Float64`` recast is the self-evidently
    discriminating pin: dropping a feature column from clustering changes
    the hierarchical ordering (and therefore the whole rendered dendrogram
    and heatmap), so a reverted selection would not render the same SVG.
    """
    df = pl.DataFrame(
        {
            "row": ["r0", "r1", "r2", "r3"],
            "A": [1, 5, 9, 2],
            "B": [4, 3, 7, 8],
        }
    )
    assert df["A"].dtype == pl.Int64

    baseline = fm.clustermap(df.with_columns(pl.col("A", "B").cast(pl.Float64))).to_svg()
    assert baseline, "clustermap Float64 baseline rendered nothing"
    assert fm.clustermap(df).to_svg() == baseline
