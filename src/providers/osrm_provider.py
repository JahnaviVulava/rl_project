"""OSRM road routing with an explicit Haversine fallback."""
from __future__ import annotations
import requests
from src.utils.geo import haversine_km

def route(origin:tuple[float,float],destination:tuple[float,float],config:dict)->dict:
    lat1,lon1=origin; lat2,lon2=destination
    url=f'{config["providers"]["osrm_url"]}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}'
    try:
        response=requests.get(url,params={"overview":"full","geometries":"geojson"},timeout=config["providers"]["timeout_seconds"]); response.raise_for_status()
        item=response.json()["routes"][0]
        return {"distance_km":item["distance"]/1000,"base_duration_minutes":item["duration"]/60,"geometry":item["geometry"]["coordinates"],"route_source":"osrm"}
    except (requests.RequestException,KeyError,IndexError,ValueError):
        distance=haversine_km(lat1,lon1,lat2,lon2)*1.25
        return {"distance_km":distance,"base_duration_minutes":distance/config["simulation"]["estimated_speed_kmh"]*60,"geometry":[[lon1,lat1],[lon2,lat2]],"route_source":"haversine_fallback"}
