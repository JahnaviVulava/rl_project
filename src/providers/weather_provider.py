"""Open-Meteo current conditions with deterministic offline fallback."""
from __future__ import annotations
import requests

def get_weather(latitude:float,longitude:float,config:dict)->dict:
    try:
        response=requests.get(config["providers"]["open_meteo_url"],params={"latitude":latitude,"longitude":longitude,"current":"temperature_2m,precipitation,rain,weather_code,wind_speed_10m"},timeout=config["providers"]["timeout_seconds"]); response.raise_for_status(); current=response.json()["current"]
        rain=float(current.get("rain",0)); precipitation=float(current.get("precipitation",0)); code=int(current.get("weather_code",0))
        severity=min(1.0,0.08*rain+0.04*precipitation+(0.25 if code>=51 else 0))
        condition="Heavy Rain" if severity>=.7 else "Light Rain" if severity>=.25 else "Cloudy" if code>=2 else "Clear"
        return {"temperature_c":float(current["temperature_2m"]),"rain_mm":rain,"precipitation_mm":precipitation,"weather_code":code,"wind_kmh":float(current.get("wind_speed_10m",0)),"condition":condition,"weather_severity":severity,"weather_source":"open_meteo_live"}
    except (requests.RequestException,KeyError,ValueError,TypeError):
        return {"temperature_c":28.0,"rain_mm":0.0,"precipitation_mm":0.0,"weather_code":0,"wind_kmh":5.0,"condition":"Clear","weather_severity":0.0,"weather_source":"offline_simulated_fallback"}
