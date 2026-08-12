"""Rate-conscious OpenStreetMap charging-station discovery via Overpass."""
from __future__ import annotations
import logging
import re
import requests
import pandas as pd
LOGGER=logging.getLogger(__name__)
USER_AGENT = "SmartCharge-RL-University-Project/1.0"


def _osm_tariff(tags: dict) -> tuple[float, str, str]:
    """Use a numeric INR/kWh OSM charge tag when available; otherwise estimate."""
    text = str(tags.get("charge", "")).strip()
    if text and re.search(r"(?:INR|Rs\.?|₹)", text, flags=re.IGNORECASE) and "kwh" in text.lower():
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if match:
            value = float(match.group(1))
            if 5 <= value <= 100:
                return value, "openstreetmap_charge_tag", text
    return 18.0, "estimated", text

def discover_stations(latitude:float,longitude:float,config:dict,minimum:int=5)->pd.DataFrame:
    """Try a bounded 30 km live query; return quickly for official fallback."""
    urls=config["providers"].get("overpass_urls", [config["providers"]["overpass_url"]]); timeout=min(5, config["providers"]["timeout_seconds"])
    best = pd.DataFrame()
    for radius_km in (30,):
        query=f'[out:json][timeout:{timeout}];(node["amenity"="charging_station"](around:{radius_km*1000},{latitude},{longitude});way["amenity"="charging_station"](around:{radius_km*1000},{latitude},{longitude}););out center tags;'
        for url in urls:
            try:
            # Overpass public instances commonly accept encoded GET queries more
            # consistently than anonymous form POSTs. The identity is explicit
            # and Streamlit does not call this provider on widget-only reruns.
                response=requests.get(url,params={"data":query},headers={"User-Agent":USER_AGENT},timeout=timeout); response.raise_for_status()
                rows=[]
                for element in response.json().get("elements",[]):
                    tags=element.get("tags",{}); center=element.get("center",element)
                    if "lat" not in center or "lon" not in center: continue
                    connector = "Connector not published"
                    if tags.get("socket:ccs") or tags.get("socket:type2_combo"):
                        connector = "CCS Type 2"
                    elif tags.get("socket:type2"):
                        connector = "Type 2"
                    elif tags.get("socket:chademo"):
                        connector = "CHAdeMO"
                    capacity = tags.get("capacity", 2)
                    try: charger_count = max(1, int(capacity))
                    except (TypeError, ValueError): charger_count = 2
                    base_price, price_source, tariff_text = _osm_tariff(tags)
                    station_name = tags.get("name") or tags.get("brand") or tags.get("operator") or f"OSM charging point {element['id']}"
                    rows.append({"station_id":f"OSM-{element['id']}","station_name":station_name,"latitude":center["lat"],"longitude":center["lon"],"connector_type":connector,"connector_verified":connector != "Connector not published","power_kw":30.0,"number_of_chargers":charger_count,"number_of_chargers_source":"openstreetmap_tag" if "capacity" in tags else "simulated_default","base_price_inr_kwh":base_price,"base_price_source":price_source,"published_tariff_text":tariff_text,"city":tags.get("addr:city","Unknown"),"state":tags.get("addr:state","Unknown"),"data_source":"openstreetmap_live"})
                frame=pd.DataFrame(rows)
                if len(frame) > len(best): best = frame
                if len(frame)>=minimum: return frame
            except (requests.RequestException,ValueError,TypeError) as exc: LOGGER.warning("Overpass failed at %s km: %s",radius_km,exc)
    return best
