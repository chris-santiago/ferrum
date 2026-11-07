"""``displot(kind="ecdf")`` and ``mark_qq`` refuse a field-bearing ``y=`` the same way.

Both draw their own computed value on y (a rolling count, sample
quantiles), so a caller-named second variable is refused rather than
silently dropped or re-pointed. ecdf's call-time keyword and its post-hoc
``.encode()`` backstop share one voicing function
(``ferrum._encode_lowering._refuse_ecdf_field_y``), so this file pins
their texts against each other as well as against a committed literal;
``mark_qq``'s own voicing is pinned the same way elsewhere
(``tests/test_statistic_bin_aggregate_refusal.py``) and exercised again
here only for the row-count-independence case.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm

_DF = pl.DataFrame({"v": list(range(1, 9)), "w": list(range(11, 19))})

_ECDF_TEXT = (
    "displot(kind='ecdf', y='w') is not supported -- kind='ecdf' draws the "
    "cumulative count of the variable x= names on y, so a second variable "
    "has nowhere to go. Drop y=, or style the count axis with "
    "fm.Y(title=..., scale=...) and no field."
)

_QQ_TEXT = (
    "mark_qq(y='w') is not supported -- mark_qq draws the sample quantiles "
    "of x= on y, so a second variable has nowhere to go. Drop y=, or style "
    "the axis with fm.Y(title=...) and no field."
)


def test_displot_ecdf_call_time_and_post_hoc_refuse_with_the_same_text():
    """The call-time ``y=`` keyword and the post-hoc ``.encode(y=...)``
    backstop raise byte-identical text for the same field."""
    with pytest.raises(ValueError) as call_time:
        fm.displot(_DF, x="v", y="w", kind="ecdf")
    with pytest.raises(ValueError) as post_hoc:
        fm.displot(_DF, x="v", kind="ecdf").encode(y="w").to_svg()
    assert str(call_time.value) == _ECDF_TEXT
    assert str(post_hoc.value) == _ECDF_TEXT


def test_displot_ecdf_wrong_type_y_is_named_by_its_stringified_value():
    """A non-channel, non-string ``y=`` (an ``int``) is named by its
    stringified value rather than silently treated as fieldless."""
    with pytest.raises(ValueError) as excinfo:
        fm.displot(_DF, x="v", y=5, kind="ecdf")
    assert str(excinfo.value) == (
        "displot(kind='ecdf', y='5') is not supported -- kind='ecdf' draws the "
        "cumulative count of the variable x= names on y, so a second variable "
        "has nowhere to go. Drop y=, or style the count axis with "
        "fm.Y(title=..., scale=...) and no field."
    )


def test_displot_ecdf_shorthand_y_is_named_by_its_whole_string():
    """A call-time ``y=`` that looks like shorthand (``"w:Q"``) is not
    parsed into a channel; the whole string is named, as any other column
    name would be."""
    with pytest.raises(ValueError) as excinfo:
        fm.displot(_DF, x="v", y="w:Q", kind="ecdf")
    assert str(excinfo.value) == (
        "displot(kind='ecdf', y='w:Q') is not supported -- kind='ecdf' draws the "
        "cumulative count of the variable x= names on y, so a second variable "
        "has nowhere to go. Drop y=, or style the count axis with "
        "fm.Y(title=..., scale=...) and no field."
    )


@pytest.mark.parametrize("n_rows", [0, 1, 8], ids=["zero-row", "one-row", "eight-row"])
def test_the_refusal_reads_bindings_not_rows(n_rows: int):
    """ecdf and ``mark_qq`` refuse a field-bearing ``y=`` identically
    whether the frame has zero, one, or many rows -- the refusal reads the
    binding, not the data."""
    frame = _DF.head(n_rows)
    with pytest.raises(ValueError) as ecdf_err:
        fm.displot(frame, x="v", y="w", kind="ecdf")
    assert str(ecdf_err.value) == _ECDF_TEXT

    with pytest.raises(ValueError) as qq_err:
        fm.Chart(frame).mark_qq().encode(x="v", y="w").to_svg()
    assert str(qq_err.value) == _QQ_TEXT
