"""Estimated charging tariff; never presented as a live tariff."""
import numpy as np


def estimate_price(base_price: float, hour: int, utilization: float, demand: float, config: dict) -> float:
    """Price = base × peak × utilization × demand, bounded by YAML limits."""
    if base_price <= 0: raise ValueError("Base price must be positive")
    peak_factor = 1 + (config["peak_markup"] if (8 <= hour < 11 or 17 <= hour < 21) else 0)
    value = base_price * peak_factor * (1 + config["utilization_markup"]*utilization) * (1 + config["demand_markup"]*demand)
    return float(np.clip(value, config["min_inr_kwh"], config["max_inr_kwh"]))
