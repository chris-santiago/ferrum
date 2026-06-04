"""Theme(color_scheme=...) validates the name against the palette registry."""

import polars as pl
import pytest

import ferrum as fm


def test_unknown_scheme_raises() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y="y")
        .theme(fm.Theme(color_scheme="not-a-real-scheme"))
    )
    with pytest.raises(ValueError) as excinfo:
        chart.to_svg()
    msg = str(excinfo.value)
    assert "Unknown color_scheme" in msg
    assert "not-a-real-scheme" in msg


@pytest.mark.parametrize(
    "name",
    [
        "okabe_ito",
        "tableau10",
        "set1",
        "set2",
        "paired",
        "pastel",
        "dark2",
        "viridis",
        "plasma",
        "magma",
        "inferno",
        "cividis",
    ],
)
def test_known_scheme_accepted(name: str) -> None:
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    svg = fm.Chart(df).mark_point().encode(x="x", y="y").theme(fm.Theme(color_scheme=name)).to_svg()
    assert svg.startswith("<svg") or svg.lstrip().startswith("<svg")
