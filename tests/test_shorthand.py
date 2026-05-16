from ferrum._shorthand import parse_shorthand


def test_bare_field():
    assert parse_shorthand("price") == ("price", None, None)


def test_field_with_type():
    assert parse_shorthand("price:Q") == ("price", "Q", None)
    assert parse_shorthand("year:T") == ("year", "T", None)
    assert parse_shorthand("species:N") == ("species", "N", None)
    assert parse_shorthand("rank:O") == ("rank", "O", None)


def test_aggregate_with_field():
    assert parse_shorthand("mean(price)") == ("price", None, "mean")
    assert parse_shorthand("median(latency)") == ("latency", None, "median")
    assert parse_shorthand("q50(latency)") == ("latency", None, "q50")


def test_aggregate_without_field():
    assert parse_shorthand("count()") == (None, None, "count")


def test_aggregate_with_field_and_type():
    assert parse_shorthand("mean(price):Q") == ("price", "Q", "mean")
    assert parse_shorthand("count():Q") == (None, "Q", "count")


def test_field_name_with_underscores_and_digits():
    assert parse_shorthand("col_42") == ("col_42", None, None)
    assert parse_shorthand("mean(col_42):Q") == ("col_42", "Q", "mean")


def test_invalid_type_raises():
    import pytest

    with pytest.raises(ValueError, match="unknown type"):
        parse_shorthand("price:Z")


def test_unbalanced_parens_raises():
    import pytest

    with pytest.raises(ValueError, match="unbalanced"):
        parse_shorthand("mean(price")
