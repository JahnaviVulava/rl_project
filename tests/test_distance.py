import pytest
from src.utils.geo import haversine_km,safe_range_km
def test_distance_nonnegative_and_symmetric():
    forward=haversine_km(10,76,11,77); reverse=haversine_km(11,77,10,76)
    assert forward>=0 and forward==pytest.approx(reverse)
def test_safe_range_formula(): assert safe_range_km(50,60,.2,.8)==pytest.approx(120)
