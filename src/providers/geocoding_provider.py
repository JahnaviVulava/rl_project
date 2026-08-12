"""OpenStreetMap Nominatim geocoding with a clear, rate-conscious identity."""
from __future__ import annotations

import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "SmartCharge-RL-University-Project/1.0"


def geocode_location(query: str, timeout_seconds: int = 10) -> dict:
    """Resolve a human location to coordinates.

    Nominatim requires a descriptive User-Agent. The Streamlit layer caches the
    returned result so repeated reruns do not repeatedly contact the service.
    """
    cleaned = query.strip()
    if len(cleaned) < 3:
        raise ValueError("Enter a more specific location, such as a neighbourhood, landmark and city")
    response = requests.get(
        NOMINATIM_URL,
        params={"q": cleaned, "format": "jsonv2", "limit": 1, "countrycodes": "in"},
        headers={"User-Agent": USER_AGENT},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    matches = response.json()
    if not matches:
        raise ValueError(f"Location not found: {cleaned}")
    match = matches[0]
    return {
        "latitude": float(match["lat"]),
        "longitude": float(match["lon"]),
        "display_name": str(match["display_name"]),
        "source": "nominatim_openstreetmap",
    }
