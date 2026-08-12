"""End-to-end live recommendation pipeline used by the Streamlit navigation UI."""
from __future__ import annotations

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo
from typing import Any
import zlib

import numpy as np
import pandas as pd

from src.environment.reward import calculate_reward
from src.providers.osm_provider import discover_stations
from src.providers.official_station_provider import discover_official_stations
from src.providers.nominatim_station_provider import discover_stations_nominatim
from src.providers.osrm_provider import route
from src.providers.weather_provider import get_weather
from src.rl.inference import recommend
from src.simulation.ev_generator import EVRequest
from src.simulation.queue_manager import StationQueue
from src.simulation.pricing import estimate_price
from src.simulation.traffic import congestion_label, estimate_traffic
from src.utils.config import project_path
from src.utils.geo import filter_candidates, haversine_km, safe_range_km

INDIA_TIMEZONE = ZoneInfo("Asia/Kolkata")


def _prepare_station_source(env, latitude: float, longitude: float, connector: str, safe_range: float) -> tuple[pd.DataFrame, str]:
    """Use current OSM data, official GHMC locations, then the supplied dataset."""
    osm = discover_stations(latitude, longitude, env.config, minimum=4)
    if not osm.empty:
        osm_candidates = filter_candidates(osm, latitude, longitude, connector, safe_range, env.k)
        if len(osm_candidates) >= 4:
            return osm_candidates, "openstreetmap_live"
        # Many valid OSM locations omit socket tags. Keep them as explicitly
        # unverified alternatives so arbitrary Indian locations do not fail.
        osm_with_distance = osm.copy()
        osm_with_distance["distance_km"] = [
            haversine_km(latitude, longitude, float(row.latitude), float(row.longitude))
            for row in osm_with_distance.itertuples()
        ]
        unverified = osm_with_distance[
            (~osm_with_distance["connector_verified"].fillna(False))
            & (osm_with_distance["distance_km"] <= safe_range)
        ]
        combined_osm = pd.concat([osm_candidates, unverified], ignore_index=True, sort=False)
        combined_osm = combined_osm.drop_duplicates("station_id").sort_values("distance_km").head(env.k)
        if len(combined_osm) >= 2:
            return combined_osm.reset_index(drop=True), "openstreetmap_live_with_unverified_connectors"

    nominatim = discover_stations_nominatim(
        latitude, longitude, env.config["providers"].get("timeout_seconds", 8)
    )
    if not nominatim.empty:
        reachable = nominatim[nominatim["distance_km"] <= safe_range].head(env.k)
        if len(reachable) >= 2:
            return reachable.reset_index(drop=True), "openstreetmap_nominatim_live"

    # Official GHMC coordinates fill the Hyderabad/Kukatpally gap in the
    # supplied national CSV. The directory does not publish connector details,
    # power, live availability or tariff, so those fields remain labelled.
    official = discover_official_stations(latitude, longitude, min(safe_range, 35.0))
    if not official.empty:
        verified_osm = osm_candidates if not osm.empty else pd.DataFrame()
        combined = pd.concat([verified_osm, official], ignore_index=True, sort=False)
        combined = combined.drop_duplicates("station_id").sort_values("distance_km").head(env.k)
        return combined.reset_index(drop=True), "live_osm_plus_ghmc_official"

    local = env.stations.copy()
    if "data_source" not in local:
        local["data_source"] = "local_dataset_fallback"
    return filter_candidates(local, latitude, longitude, connector, safe_range, env.k), "local_dataset_fallback"


def _ensure_queue_state(env, candidates: pd.DataFrame) -> None:
    """Add stations and seed a reproducible simulated snapshot once per station."""
    initialized = getattr(env, "_initialized_live_station_ids", set())
    for row in candidates.itertuples():
        station_id = str(row.station_id)
        if station_id not in env.queue_manager.stations:
            env.queue_manager.stations[station_id] = StationQueue(max(1, int(row.number_of_chargers)))
        if station_id not in initialized:
            state = env.queue_manager.stations[station_id]
            fingerprint = zlib.crc32(station_id.encode("utf-8"))
            occupied = fingerprint % (state.total_chargers + 1)
            state.active_sessions = [float(12 + ((fingerprint >> (i * 3)) % 42)) for i in range(occupied)]
            if occupied == state.total_chargers:
                state.waiting_durations = [float(24 + fingerprint % 35)] * int((fingerprint // 7) % 3)
            initialized.add(station_id)
    env._initialized_live_station_ids = initialized


def _estimated_base_price(station_id: str, power_kw: float) -> float:
    """Stable, visibly estimated station tariff input; never a live price."""
    variation = (zlib.crc32(station_id.encode("utf-8")) % 7 - 3) * 0.35
    return float(np.clip(14.0 + 0.055 * power_kw + variation, 12.0, 24.0))


def _preference_weights(preference: str) -> dict[str, float]:
    return {
        "Fastest": {"time": .70, "cost": .10, "wait": .15, "distance": .05},
        "Lowest Waiting": {"time": .20, "cost": .10, "wait": .65, "distance": .05},
        "Cheapest": {"time": .20, "cost": .65, "wait": .10, "distance": .05},
        "Balanced": {"time": .45, "cost": .30, "wait": .15, "distance": .10},
    }.get(preference, {"time": .45, "cost": .30, "wait": .15, "distance": .10})


def _add_decision_scores(rows: list[dict], preference: str) -> None:
    """Add a transparent relative 0-100 time/cost/wait/distance score."""
    weights = _preference_weights(preference)
    for field, output in (("total_minutes", "time_rank_score"), ("estimated_cost_inr", "cost_rank_score"),
                          ("wait_minutes", "wait_rank_score"), ("distance_km", "distance_rank_score")):
        values = np.asarray([row[field] for row in rows], dtype=float)
        low, high = float(values.min()), float(values.max())
        scores = np.full(len(rows), 100.0) if abs(high - low) < 1e-9 else 100.0 * (high - values) / (high - low)
        for row, score in zip(rows, scores):
            row[output] = float(score)
    for row in rows:
        row["decision_score"] = float(
            weights["time"] * row["time_rank_score"] + weights["cost"] * row["cost_rank_score"]
            + weights["wait"] * row["wait_rank_score"] + weights["distance"] * row["distance_rank_score"]
        )


def build_recommendation(env, request: EVRequest, location_name: str) -> dict[str, Any]:
    """Run discovery → routing → weather → PPO and return stable display data."""
    config = env.config
    simulation = config["simulation"]
    env.request = request
    env.safe_range = safe_range_km(
        request.soc_percent,
        request.battery_capacity_kwh,
        simulation["consumption_kwh_km"],
        simulation["safe_range_factor"],
    )
    candidates, station_source = _prepare_station_source(
        env, request.latitude, request.longitude, request.connector_type, env.safe_range
    )
    if candidates.empty:
        raise ValueError("No compatible charging station is safely reachable from this location.")

    current_time = datetime.now(INDIA_TIMEZONE)
    _ensure_queue_state(env, candidates)
    candidate_records = candidates.to_dict("records")

    def fetch_route(row: dict) -> dict:
        return route(
            (request.latitude, request.longitude),
            (float(row["latitude"]), float(row["longitude"])),
            config,
        )

    # Weather and road-route retrieval are independent network calls.
    with ThreadPoolExecutor(max_workers=min(7, len(candidate_records) + 1)) as executor:
        weather_future = executor.submit(get_weather, request.latitude, request.longitude, config)
        route_results = list(executor.map(fetch_route, candidate_records))
        weather = weather_future.result()

    env.hour = current_time.hour
    env.day_of_week = current_time.weekday()
    env.weather_severity = float(weather["weather_severity"])
    env.temperature_c = float(weather["temperature_c"])
    env.rain_mm = float(weather["rain_mm"])
    env.traffic = estimate_traffic(env.hour, env.day_of_week, env.weather_severity, env.np_random)

    routed_rows: list[dict] = []
    for row, route_data in zip(candidate_records, route_results):
        if route_data["distance_km"] > env.safe_range:
            continue
        # DERIVED: route-specific congestion adds small bounded distance and
        # station-load effects to the shared time/weather traffic estimate.
        station_state = env.queue_manager.stations.get(str(row["station_id"]))
        utilization = station_state.utilization if station_state else 0.0
        road_congestion_score = float(np.clip(
            env.traffic + 0.06 * min(route_data["distance_km"] / 30.0, 1.0) + 0.05 * utilization,
            0.0, 1.0,
        ))
        row.update({
            "distance_km": float(route_data["distance_km"]),
            "road_distance_km": float(route_data["distance_km"]),
            "base_route_duration_minutes": float(route_data["base_duration_minutes"]),
            "route_geometry": route_data["geometry"],
            "route_source": route_data["route_source"],
            "road_congestion_score": road_congestion_score,
            "base_price_inr_kwh": float(row["base_price_inr_kwh"]) if row.get("base_price_source") == "openstreetmap_charge_tag" else _estimated_base_price(str(row["station_id"]), float(row["power_kw"])),
        })
        routed_rows.append(row)

    env.candidates = pd.DataFrame(routed_rows).sort_values("road_distance_km").head(env.k).reset_index(drop=True)
    if env.candidates.empty:
        raise ValueError("Stations were discovered, but none is reachable by road within the safe range.")
    env.candidate_mask = np.zeros(env.k, dtype=bool)
    env.candidate_mask[:len(env.candidates)] = True

    raw_ppo_index, model_source = recommend(env, project_path("models/ppo_ev_charging"), "PPO")
    network_utilizations = [state.utilization for state in env.queue_manager.stations.values()]
    rows: list[dict[str, Any]] = []
    reward_components: dict[str, dict] = {}
    for index, station in env.candidates.iterrows():
        metrics = env._candidate_metrics(station)
        connector_verified = bool(station.get("connector_verified", True))
        metrics["compatible"] = connector_verified
        metrics["utilization_std"] = float(np.std(network_utilizations))
        score, components = calculate_reward(metrics, config["reward"])
        state = env.queue_manager.stations[str(station.station_id)]
        off_peak_price = estimate_price(
            float(station.base_price_inr_kwh), 12, metrics["utilization"], metrics["future_demand"], config["pricing"]
        )
        item = {
            "index": int(index),
            "station_id": str(station.station_id),
            "station_name": str(station.station_name),
            "latitude": float(station.latitude),
            "longitude": float(station.longitude),
            "connector_type": str(station.connector_type),
            "connector_verified": connector_verified,
            "station_source": str(station.get("data_source", station_source)),
            "route_source": str(station.route_source),
            "route_geometry": station.route_geometry,
            "distance_km": metrics["distance_km"],
            "base_travel_minutes": metrics["base_travel_minutes"],
            "road_congestion_score": metrics["road_congestion_score"],
            "road_congestion_label": congestion_label(metrics["road_congestion_score"]),
            "congestion_delay_minutes": metrics["travel_minutes"] - metrics["base_travel_minutes"],
            "weather_delay_minutes": metrics["base_travel_minutes"] * 0.75 * 0.20 * env.weather_severity,
            "travel_minutes": metrics["travel_minutes"],
            "wait_minutes": metrics["wait_minutes"],
            "charge_minutes": metrics["charge_minutes"],
            "total_minutes": metrics["travel_minutes"] + metrics["wait_minutes"] + metrics["charge_minutes"],
            "power_kw": float(station.power_kw),
            "free_chargers": state.free_chargers,
            "total_chargers": state.total_chargers,
            "occupied_chargers": state.occupied_chargers,
            "queue_length": state.queue_length,
            "active_remaining_minutes": list(state.active_sessions),
            "price_inr_kwh": metrics["price_inr_kwh"],
            "base_price_inr_kwh": float(station.base_price_inr_kwh),
            "price_source": str(station.get("base_price_source", "estimated")),
            "published_tariff_text": str(station.get("published_tariff_text", "")),
            "off_peak_price_inr_kwh": off_peak_price,
            "peak_surcharge_inr_kwh": max(0.0, metrics["price_inr_kwh"] - off_peak_price),
            "peak_cost_impact_inr": max(0.0, metrics["price_inr_kwh"] - off_peak_price) * request.required_energy_kwh,
            "estimated_cost_inr": metrics["estimated_cost_inr"],
            "future_demand": metrics["future_demand"],
            "station_utilization": metrics["utilization"],
            "score": score,
        }
        rows.append(item)
        reward_components[str(index)] = components

    _add_decision_scores(rows, request.preference)

    # A trained policy may be imperfect. Never present a clearly dominated PPO
    # action as the final recommendation: use a transparent time/cost guardrail.
    guardrail = config.get("recommendation_guardrail", {})
    maximum_extra = float(guardrail.get("maximum_extra_minutes", 8.0))
    maximum_cost_fraction = float(guardrail.get("maximum_extra_cost_fraction", 0.10))
    verified = [item for item in rows if item["connector_verified"]]
    eligible = verified or rows
    best_current = max(eligible, key=lambda item: (item["decision_score"], -item["total_minutes"], -item["estimated_cost_inr"]))
    raw_choice = rows[raw_ppo_index]
    minimum_score_improvement = float(guardrail.get("minimum_decision_score_improvement", 1.0))
    preference_improvement = best_current["decision_score"] > raw_choice["decision_score"] + minimum_score_improvement
    dominated = preference_improvement or (
        raw_choice["total_minutes"] > best_current["total_minutes"] + maximum_extra
        and best_current["estimated_cost_inr"] <= raw_choice["estimated_cost_inr"] * (1 + maximum_cost_fraction)
    ) or (not raw_choice["connector_verified"] and bool(verified))
    recommended_index = int(best_current["index"] if dominated else raw_ppo_index)
    decision_source = "ppo_with_guardrail_override" if dominated else model_source

    observation = env._observation().tolist()
    return {
        "rows": rows,
        "recommended_index": recommended_index,
        "selected_index": recommended_index,
        "model_source": model_source,
        "decision_source": decision_source,
        "raw_ppo_index": int(raw_ppo_index),
        "guardrail_applied": bool(dominated),
        "location": (request.latitude, request.longitude),
        "location_name": location_name,
        "required_energy_kwh": request.required_energy_kwh,
        "safe_range_km": env.safe_range,
        "weather": weather,
        "current_time_iso": current_time.isoformat(),
        "current_time_display": current_time.strftime("%A · %I:%M %p"),
        "road_traffic_score": env.traffic,
        "preference": request.preference,
        "is_peak_hour": bool(8 <= env.hour < 11 or 17 <= env.hour < 21),
        "pricing_period": "Peak" if (8 <= env.hour < 11 or 17 <= env.hour < 21) else "Off-peak",
        "peak_markup_percent": float(config["pricing"]["peak_markup"] * 100),
        "station_source": station_source,
        "ppo_observation": observation,
        "station_id_mapping": {str(i): row["station_id"] for i, row in enumerate(rows)},
        "reward_components": reward_components,
    }


def recommendation_explanation(result: dict) -> str:
    """Explain the recommendation using only computed values."""
    recommended = result["rows"][result["recommended_index"]]
    nearest = min(result["rows"], key=lambda item: item["distance_km"])
    if result.get("guardrail_applied"):
        raw = result["rows"][result["raw_ppo_index"]]
        return (
            f"The PPO proposed {raw['station_name']} ({raw['total_minutes']:.0f} minutes), but the quality "
            f"guardrail selected {recommended['station_name']} using the {result.get('preference', 'Balanced')} "
            f"preference. Its estimated total is {recommended['total_minutes']:.0f} minutes and charging cost is "
            f"₹{recommended['estimated_cost_inr']:.0f}. Time, waiting, electricity cost and distance all participate "
            f"in the final decision."
        )
    if nearest["station_id"] == recommended["station_id"]:
        return (
            f"{recommended['station_name']} is both the nearest compatible reachable station and the "
            f"model choice. Its estimated total time is {recommended['total_minutes']:.0f} minutes, "
            f"with {recommended['free_chargers']} of {recommended['total_chargers']} chargers free."
        )
    extra_travel = recommended["travel_minutes"] - nearest["travel_minutes"]
    wait_saved = nearest["wait_minutes"] - recommended["wait_minutes"]
    total_saved = nearest["total_minutes"] - recommended["total_minutes"]
    if total_saved < 0:
        return (
            f"PPO selected {recommended['station_name']}, but the current deterministic estimate is "
            f"{abs(total_saved):.0f} minutes longer than the nearest compatible option, "
            f"{nearest['station_name']}. Both currently have an estimated {recommended['wait_minutes']:.0f}-minute "
            f"wait. This is an honest model-quality warning: the driver should review the alternatives, and this "
            f"scenario should be included in future PPO retraining."
        )
    return (
        f"{nearest['station_name']} is {nearest['distance_km']:.1f} km away, but has an estimated "
        f"{nearest['wait_minutes']:.0f}-minute wait and {nearest['free_chargers']} free chargers. "
        f"The recommended {recommended['station_name']} adds {max(0, extra_travel):.0f} minutes of road travel "
        f"but changes expected waiting by {wait_saved:.0f} minutes. Its computed total time is "
        f"{recommended['total_minutes']:.0f} minutes ({total_saved:.0f} minutes lower than the nearest station)."
    )
