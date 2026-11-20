from _common import registry


def test_overview_has_grounded_city_location_counts_and_dashboard_map():
    from cortexhr_phase4.services.handlers import overview
    from cortexhr_phase4.console.dashboard import build
    data = overview(registry())
    locations = data["hiring"]["location_counts"]
    assert data["hiring"]["active_employees"] == 430
    assert len(locations) == 9
    assert sum(x["employees"] for x in locations) == 430
    assert all(x["coordinate_type"] == "approximate city centre" for x in locations)
    html = build(data, [], {"name": "QA", "role": "Admin"}, mode="embed", env="development")
    assert 'id="cx-map"' in html
    assert "hiring_map_reference.png" not in html
