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

    with pytest.raises(ValueError, match=r"parse_shorthand: type must be one of"):
        parse_shorthand("price:Z")


def test_unbalanced_parens_raises():
    import pytest

    with pytest.raises(ValueError, match="unbalanced"):
        parse_shorthand("mean(price")


# --- Permissive field-name tests (fix/shorthand-hyphen-parsing) ---


def test_bare_field_with_hyphens():
    assert parse_shorthand("raw-aucpr") == ("raw-aucpr", None, None)


def test_field_with_hyphens_and_type():
    assert parse_shorthand("raw-aucpr:Q") == ("raw-aucpr", "Q", None)


def test_field_with_dots():
    assert parse_shorthand("model.score") == ("model.score", None, None)
    assert parse_shorthand("model.score:Q") == ("model.score", "Q", None)


def test_field_with_spaces():
    assert parse_shorthand("Sepal Length") == ("Sepal Length", None, None)
    assert parse_shorthand("Sepal Length:Q") == ("Sepal Length", "Q", None)


def test_field_with_mixed_special_chars():
    assert parse_shorthand("col-1.2 foo") == ("col-1.2 foo", None, None)
    assert parse_shorthand("col-1.2 foo:N") == ("col-1.2 foo", "N", None)


def test_aggregate_with_hyphenated_field():
    assert parse_shorthand("mean(raw-aucpr)") == ("raw-aucpr", None, "mean")
    assert parse_shorthand("mean(raw-aucpr):Q") == ("raw-aucpr", "Q", "mean")


def test_aggregate_with_dotted_field():
    assert parse_shorthand("sum(model.score)") == ("model.score", None, "sum")


def test_aggregate_with_spaced_field():
    assert parse_shorthand("mean(Sepal Length)") == ("Sepal Length", None, "mean")


def test_colon_in_column_name_informative_error():
    """Column names containing ':' should produce a helpful error message."""
    import pytest

    with pytest.raises(ValueError, match=r"looks like.*type suffix"):
        parse_shorthand("col:name:Q")

    with pytest.raises(ValueError, match=r"looks like.*type suffix"):
        parse_shorthand("ns:field")

    with pytest.raises(ValueError, match=r"looks like.*type suffix"):
        parse_shorthand("a:b:c")
