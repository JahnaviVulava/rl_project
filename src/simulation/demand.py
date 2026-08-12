"""Bounded, interpretable near-future demand estimator."""
import numpy as np


def estimate_future_demand(hour: int, recent_arrivals: int, traffic: float, weather_severity: float, utilization: float) -> float:
    """Combine normalized evidence; coefficients sum to one."""
    time_signal = 1.0 if (8 <= hour < 11 or 17 <= hour < 21) else 0.35
    arrivals = min(max(recent_arrivals / 10.0, 0.0), 1.0)
    return float(np.clip(0.25*time_signal + 0.25*arrivals + 0.20*traffic + 0.10*weather_severity + 0.20*utilization, 0, 1))
