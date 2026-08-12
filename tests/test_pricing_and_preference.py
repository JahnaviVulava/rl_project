import pytest

from src.providers.osm_provider import _osm_tariff
from src.services.recommendation_service import _add_decision_scores
from src.simulation.pricing import estimate_price
from src.utils.config import load_config


def test_peak_hour_price_is_higher_than_off_peak():
    config = load_config()["pricing"]
    off_peak = estimate_price(18.0, 12, 0.4, 0.5, config)
    peak = estimate_price(18.0, 18, 0.4, 0.5, config)
    assert peak > off_peak
    assert peak / off_peak == pytest.approx(1 + config["peak_markup"])


def test_preference_changes_final_decision_score():
    rows = [
        {"total_minutes": 35.0, "estimated_cost_inr": 650.0, "wait_minutes": 2.0, "distance_km": 4.0},
        {"total_minutes": 48.0, "estimated_cost_inr": 390.0, "wait_minutes": 5.0, "distance_km": 6.0},
    ]
    _add_decision_scores(rows, "Fastest")
    assert rows[0]["decision_score"] > rows[1]["decision_score"]
    _add_decision_scores(rows, "Cheapest")
    assert rows[1]["decision_score"] > rows[0]["decision_score"]


def test_osm_published_inr_tariff_is_used_when_parseable():
    value, source, text = _osm_tariff({"charge": "INR 19.5/kWh"})
    assert value == 19.5
    assert source == "openstreetmap_charge_tag"
    assert text == "INR 19.5/kWh"
