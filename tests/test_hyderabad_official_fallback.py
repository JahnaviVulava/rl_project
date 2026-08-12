from src.providers.official_station_provider import discover_official_stations


def test_kukatpally_has_multiple_official_fallback_locations():
    stations = discover_official_stations(17.4930841, 78.4054408, radius_km=8.0)
    assert len(stations) >= 4
    assert stations.station_name.str.contains("Kukatpally|KPHB", case=False).any()
    assert stations.data_source.eq("ghmc_official_directory").all()
    assert not stations.connector_verified.any()
