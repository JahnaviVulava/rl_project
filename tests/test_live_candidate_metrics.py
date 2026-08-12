import pandas as pd
import pytest

from src.environment.ev_charging_env import EVChargingEnv


def test_offline_candidate_without_route_congestion_uses_environment_fallback():
    """Regression: pandas Series must not raise KeyError after env.step regenerates candidates."""
    env = EVChargingEnv()
    env.reset(seed=42)
    row = env.candidates.iloc[0].copy()
    row = row.drop(labels=[name for name in ("road_congestion_score", "road_distance_km", "base_route_duration_minutes") if name in row.index])
    metrics = env._candidate_metrics(row)
    assert metrics["road_congestion_score"] == env.traffic
    assert metrics["distance_km"] >= 0
    assert metrics["travel_minutes"] >= 0


def test_routed_candidate_uses_road_values():
    env = EVChargingEnv()
    env.reset(seed=42)
    row = env.candidates.iloc[0].copy()
    row["road_distance_km"] = 7.5
    row["base_route_duration_minutes"] = 12.0
    row["road_congestion_score"] = 0.4
    metrics = env._candidate_metrics(row)
    assert metrics["distance_km"] == 7.5
    assert metrics["base_travel_minutes"] == 12.0
    assert metrics["travel_minutes"] == pytest.approx(15.6)
