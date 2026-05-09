def test_chart_spec_smoke():
    from ferrum._core import ChartSpec

    spec = ChartSpec(mark="point", x="a", y="b")
    json = spec.to_json()
    spec2 = ChartSpec.from_json(json)
    assert spec == spec2
    assert spec.to_json() == spec2.to_json()
