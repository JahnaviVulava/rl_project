"""Interpretable estimated congestion, explicitly not live traffic."""
from __future__ import annotations
import numpy as np


def estimate_traffic(hour: int, day_of_week: int, weather_severity: float, rng: np.random.Generator | None = None) -> float:
    """Return congestion in [0,1] from time, weekday, weather and small incident noise."""
    if not 0 <= hour <= 23 or not 0 <= day_of_week <= 6: raise ValueError("Invalid time")
    severity = float(np.clip(weather_severity, 0, 1))
    peak = 0.35 if (8 <= hour < 11 or 17 <= hour < 21) else 0.08
    weekday = 0.12 if day_of_week < 5 else 0.04
    incident = float((rng or np.random.default_rng()).uniform(0, 0.08))
    return float(np.clip(0.12 + peak + weekday + 0.20 * severity + incident, 0, 1))


def congestion_label(level: float) -> str:
    if level < 0.25: return "Low"
    if level < 0.50: return "Moderate"
    if level < 0.75: return "High"
    return "Severe"
