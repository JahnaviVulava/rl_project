"""Geographic calculations and candidate filtering."""
from __future__ import annotations
import math
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in kilometres; always non-negative."""
    values = (lat1, lon1, lat2, lon2)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Coordinates must be finite")
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi, d_lambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a)))


def safe_range_km(soc_percent: float, capacity_kwh: float, consumption_kwh_km: float = 0.18, safety_factor: float = 0.8) -> float:
    """Compute ``SOC × capacity / consumption × safety factor``."""
    if not 0 <= soc_percent <= 100 or capacity_kwh <= 0 or consumption_kwh_km <= 0 or not 0 < safety_factor <= 1:
        raise ValueError("Invalid battery or safety parameters")
    return (soc_percent / 100.0) * capacity_kwh / consumption_kwh_km * safety_factor


def filter_candidates(stations: pd.DataFrame, latitude: float, longitude: float, connector: str, safe_range: float, k: int) -> pd.DataFrame:
    """Filter compatible, reachable stations and return the nearest ``k``."""
    required = {"latitude", "longitude", "connector_type"}
    if not required.issubset(stations.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(stations.columns))}")
    result = stations.copy()
    result["distance_km"] = [haversine_km(latitude, longitude, row.latitude, row.longitude) for row in result.itertuples()]
    wanted = connector.strip().lower()
    compatible = result["connector_type"].str.lower().map(lambda value: wanted in value or value in wanted or wanted == "other")
    result = result[compatible & (result["distance_km"] <= safe_range)]
    return result.sort_values("distance_km").head(k).reset_index(drop=True)
