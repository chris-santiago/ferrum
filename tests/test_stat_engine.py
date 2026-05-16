"""Phase 5 smoke tests — one happy path per transform via polars DataFrames."""

import polars as pl
from ferrum._core import (
    Aggregate,
    AggregateOp,
    Bin,
    ChartSpec,
    Kde,
    Smooth,
    Summary,
)


def test_bin_smoke():
    spec = Bin(field="price", bin_count=10)
    cs = ChartSpec(mark="bar", x="price", transforms=[spec])
    assert cs.transforms_len == 1 if hasattr(cs, "transforms_len") else len(cs.transforms) == 1


def test_kde_smoke():
    spec = Kde(field="price", bandwidth="scott", n=128)
    cs = ChartSpec(mark="line", x="price", transforms=[spec])
    rt = ChartSpec.from_json(cs.to_json())
    assert rt == cs


def test_smooth_lm_smoke():
    spec = Smooth(x="x", y="y", method="lm", ci=0.95, n=50)
    cs = ChartSpec(mark="line", x="x", y="y", transforms=[spec])
    rt = ChartSpec.from_json(cs.to_json())
    assert rt == cs


def test_smooth_loess_smoke():
    spec = Smooth(x="x", y="y", method="loess", bandwidth=0.5, degree=2, n=50, seed=42)
    cs = ChartSpec(mark="line", x="x", y="y", transforms=[spec])
    rt = ChartSpec.from_json(cs.to_json())
    assert rt == cs


def test_aggregate_smoke():
    spec = Aggregate(
        ops=[
            AggregateOp("price", "mean", "avg_price"),
            AggregateOp("price", "median", "med_price"),
        ],
        groupby=["region"],
    )
    cs = ChartSpec(mark="bar", x="region", transforms=[spec])
    rt = ChartSpec.from_json(cs.to_json())
    assert rt == cs


def test_summary_smoke():
    spec = Summary(field="latency", groupby=["service"], error_fn="ci", n_boot=200, seed=0)
    cs = ChartSpec(mark="rule", x="service", transforms=[spec])
    rt = ChartSpec.from_json(cs.to_json())
    assert rt == cs


def test_full_pipeline_round_trip():
    cs = ChartSpec(
        mark="point",
        x="x",
        transforms=[
            Bin(field="x", bin_count=8),
            Aggregate(
                ops=[AggregateOp("count", "sum", "total")],
                groupby=["bin_start"],
            ),
        ],
    )
    j = cs.to_json()
    rt = ChartSpec.from_json(j)
    assert rt == cs
    assert len(rt.transforms) == 2


def test_dataframe_acceptance_smoke():
    # Constructing a chart spec doesn't actually apply transforms; that's the engine's job.
    # Just confirm the pyclasses don't choke on a typical polars-DataFrame field-name workflow.
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 4.0, 9.0]})
    fields = df.columns
    assert "x" in fields
    spec = Smooth(x="x", y="y", method="lm")
    cs = ChartSpec(mark="line", x="x", y="y", transforms=[spec])
    assert "x" in cs.to_json()
