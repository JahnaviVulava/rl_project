"""Official municipal station-directory fallback for Hyderabad."""
from __future__ import annotations

import pandas as pd

from src.utils.config import project_path
from src.utils.geo import haversine_km


def discover_official_stations(latitude: float, longitude: float, radius_km: float = 35.0) -> pd.DataFrame:
    """Return nearby GHMC-listed locations; equipment fields remain explicitly estimated."""
    path = project_path("data/raw/GHMC_EV_Stations_Official.csv")
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["distance_km"] = [
        haversine_km(latitude, longitude, float(row.latitude), float(row.longitude))
        for row in frame.itertuples()
    ]
    frame = frame[frame["distance_km"] <= radius_km].copy()
    frame["connector_type"] = "Connector not published"
    frame["connector_verified"] = False
    frame["power_kw"] = 30.0
    frame["number_of_chargers"] = 2
    frame["number_of_chargers_source"] = "simulated_default"
    frame["base_price_inr_kwh"] = 18.0
    frame["base_price_source"] = "estimated"
    frame["data_source"] = "ghmc_official_directory"
    return frame.sort_values("distance_km").reset_index(drop=True)
