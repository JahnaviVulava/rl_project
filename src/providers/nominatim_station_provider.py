"""Bounded OpenStreetMap POI fallback when public Overpass instances are slow."""
from __future__ import annotations

import requests
import pandas as pd

from src.utils.geo import haversine_km

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "SmartCharge-RL-University-Project/1.0"


def discover_stations_nominatim(latitude: float, longitude: float, timeout_seconds: int = 8) -> pd.DataFrame:
    """Run one user-triggered, geographically bounded charging-station search."""
    delta = 0.30
    viewbox = f"{longitude-delta},{latitude+delta},{longitude+delta},{latitude-delta}"
    try:
        response = requests.get(
            NOMINATIM_URL,
            params={
                "q": "charging station",
                "format": "jsonv2",
                "limit": 25,
                "countrycodes": "in",
                "bounded": 1,
                "viewbox": viewbox,
                "namedetails": 1,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=min(8, timeout_seconds),
        )
        response.raise_for_status()
        rows = []
        for item in response.json():
            if item.get("category", item.get("class")) != "amenity" or item.get("type") != "charging_station":
                continue
            lat, lon = float(item["lat"]), float(item["lon"])
            published_name = (item.get("namedetails") or {}).get("name")
            nearby_label = str(item.get("display_name", "Charging station")).split(",")[0]
            name = published_name or f"EV charging station near {nearby_label}"
            rows.append({
                "station_id": f"NOM-{item['osm_type']}-{item['osm_id']}",
                "station_name": name,
                "latitude": lat,
                "longitude": lon,
                "connector_type": "Connector not published",
                "connector_verified": False,
                "power_kw": 30.0,
                "number_of_chargers": 2,
                "number_of_chargers_source": "simulated_default",
                "base_price_inr_kwh": 18.0,
                "base_price_source": "estimated",
                "published_tariff_text": "",
                "city": "Unknown",
                "state": "Unknown",
                "data_source": "openstreetmap_nominatim_live",
                "distance_km": haversine_km(latitude, longitude, lat, lon),
            })
        return pd.DataFrame(rows).sort_values("distance_km").reset_index(drop=True) if rows else pd.DataFrame()
    except (requests.RequestException, ValueError, TypeError, KeyError):
        return pd.DataFrame()
