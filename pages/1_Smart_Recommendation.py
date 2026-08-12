"""Professional map-first charging-planner dashboard."""
from __future__ import annotations

from datetime import datetime
from html import escape
from zoneinfo import ZoneInfo
from copy import deepcopy
from urllib.parse import urlencode

import folium
from folium import Element
from folium.plugins import Fullscreen
import streamlit as st
from streamlit_folium import st_folium

from src.environment.ev_charging_env import EVChargingEnv
from src.providers.geocoding_provider import geocode_location
from src.services.recommendation_service import build_recommendation
from src.simulation.ev_generator import EVRequest
from src.utils.config import load_config

INDIA_CENTER = (20.5937, 78.9629)

st.markdown("""
<style>
  .block-container {padding-left:1.25rem; padding-right:1.25rem; padding-bottom:2rem; max-width:1900px;}
  [data-testid="stSidebar"] {min-width:230px; max-width:230px; border-right:1px solid var(--line);}
  .brand {font-size:1.45rem; font-weight:800; letter-spacing:-.03em; margin:.25rem 0 .1rem; color:var(--text);}
  .tagline,.muted {color:var(--muted); font-size:.77rem;}
  .top-strip {display:flex; justify-content:space-between; align-items:center; gap:1rem; margin:.1rem 0 .65rem;}
  .top-weather {border:1px solid var(--line); border-radius:12px; padding:.55rem .8rem; min-width:280px; text-align:right; background:var(--panel); color:var(--text);}
  .dashboard-panel {border:1px solid var(--line); border-radius:14px; background:var(--panel); padding:.85rem;}
  .panel-title {font-size:.94rem; font-weight:750; margin-bottom:.2rem; color:var(--text);}
  .section-label {font-size:.68rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin-top:.55rem;}
  .recommendation-card {border:1px solid var(--accent); box-shadow:0 0 0 1px var(--accent-soft); border-radius:14px; padding:.9rem; background:var(--accent-soft); color:var(--text);}
  .recommendation-title {font-size:1.23rem; font-weight:800; line-height:1.22; margin:.25rem 0 .3rem; color:var(--text);}
  .score-badge {float:right; background:var(--accent-soft); color:var(--accent); border:1px solid var(--accent); border-radius:9px; padding:.42rem .55rem; font-weight:800;}
  .metric-grid {display:grid; grid-template-columns:1fr 1fr; gap:.42rem; margin:.65rem 0;}
  .metric-box {background:var(--metric-bg); border-radius:9px; padding:.5rem .55rem;}
  .metric-label {font-size:.64rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em;}
  .metric-value {font-size:1rem; font-weight:750; margin-top:.08rem; color:var(--text);}
  .total-value {color:var(--accent); font-size:1.14rem;}
  .option-card {border:1px solid var(--line); border-radius:11px; padding:.65rem .72rem; background:var(--panel); margin-top:.5rem; color:var(--text);}
  .option-name {font-weight:750; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
  .option-grid {display:grid; grid-template-columns:repeat(3,1fr); gap:.3rem; margin-top:.4rem;}
  .navigation-card {border:1px solid #38bdf8; background:rgba(56,189,248,.09); border-radius:12px; padding:.72rem; margin:.65rem 0; color:var(--text);}
  .navigation-title {font-weight:800; font-size:.94rem; margin-bottom:.2rem;}
  .difference {color:var(--muted); font-size:.74rem; margin-top:.35rem;}
  .data-note {font-size:.72rem; color:var(--muted); line-height:1.45; border-top:1px solid var(--line); padding-top:.55rem; margin-top:.5rem;}
  .status-grid {display:grid; grid-template-columns:1fr auto; gap:.42rem; font-size:.73rem; margin-top:.7rem;}
  .active {color:#4ade80; font-weight:700;}
  div[data-testid="stForm"] {border:0; padding:0;}
  div[data-testid="stMetric"] {border:1px solid var(--line); border-radius:10px; padding:.5rem .6rem; background:var(--panel); color:var(--text);}
  div[data-testid="stMetricValue"] {font-size:1.12rem;}
  .stButton button[kind="primary"], .stFormSubmitButton button {background:var(--accent); color:white; border:0; font-weight:800;}
  iframe {border-radius:14px; border:1px solid var(--line);}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_geocode(query: str) -> dict:
    return geocode_location(query)


def initialize_environment() -> EVChargingEnv:
    env = EVChargingEnv(config=load_config("live"))
    env.reset(seed=42)
    return env


if "smart_env" not in st.session_state:
    st.session_state.smart_env = initialize_environment()
if "recommendation" not in st.session_state:
    st.session_state.recommendation = None

env = st.session_state.smart_env
result = st.session_state.recommendation
now = datetime.now(ZoneInfo("Asia/Kolkata"))

weather = result["weather"] if result else None
location_short = result["location_name"].split(",")[0] if result else "Location not selected"
weather_icon = "☀️"
if weather and (weather["rain_mm"] > 0 or "rain" in weather["condition"].lower()):
    weather_icon = "🌧️"
elif weather and "cloud" in weather["condition"].lower():
    weather_icon = "☁️"
weather_summary = (
    f'<b>{weather_icon} &nbsp; {weather["temperature_c"]:.1f} C &nbsp; {escape(weather["condition"])}</b><br>'
    f'<span class="muted">Rain {weather["rain_mm"]:.1f} mm &nbsp; | &nbsp; Wind {weather.get("wind_kmh", 0):.1f} km/h &nbsp; | &nbsp; {now.strftime("%I:%M %p")}</span>'
    if weather else f'<b>Current conditions</b><br><span class="muted">Search a location &nbsp; | &nbsp; {now.strftime("%I:%M %p")}</span>'
)
def sync_theme() -> None:
    st.session_state.ui_theme = "light" if st.session_state.header_light_mode else "dark"


if "header_light_mode" not in st.session_state:
    st.session_state.header_light_mode = st.session_state.get("ui_theme", "dark") == "light"
header_title, header_weather, header_theme = st.columns([0.52, 0.34, 0.14], vertical_alignment="center")
with header_title:
    st.markdown('<div class="brand">Plan a charge</div><div class="tagline">Routes, weather, queues and charging cost in one view.</div>', unsafe_allow_html=True)
with header_weather:
    st.markdown(f'<div class="top-weather">{weather_summary}</div>', unsafe_allow_html=True)
with header_theme:
    st.toggle("Light mode", key="header_light_mode", on_change=sync_theme)

location_query = st.text_input("Search location", value="", placeholder="Enter neighbourhood, landmark or city", label_visibility="collapsed")


def create_map(active_result: dict | None) -> folium.Map:
    center = active_result["location"] if active_result else INDIA_CENTER
    light_map = st.session_state.get("ui_theme", "dark") == "light"
    map_view = folium.Map(location=center, zoom_start=13 if active_result else 5,
                          tiles="OpenStreetMap" if light_map else "CartoDB dark_matter",
                          control_scale=True, zoom_control=True)
    Fullscreen(position="topleft", title="Open full screen", title_cancel="Exit full screen").add_to(map_view)
    if not active_result:
        return map_view

    folium.Marker(active_result["location"], tooltip="Your location",
                  popup=f"<b>Your location</b><br>{escape(active_result['location_name'])}",
                  icon=folium.Icon(color="blue", icon="car", prefix="fa")).add_to(map_view)
    recommended_index = active_result["recommended_index"]
    selected_index = active_result.get("selected_index", recommended_index)
    ranked = sorted(active_result["rows"], key=lambda row: (-row.get("decision_score", 0), row["total_minutes"]))
    recommended_row = active_result["rows"][recommended_index]
    route_items = [recommended_row] + [row for row in ranked if row["station_id"] != recommended_row["station_id"]][:2]
    routed_ids = {row["station_id"] for row in route_items}
    route_palette = ["#10b981", "#f59e0b", "#ef5b4f"]
    route_colors = {row["station_id"]: route_palette[index] for index, row in enumerate(route_items)}
    # Paint lower-priority routes first so the green recommendation remains on top.
    for item in reversed(route_items):
        selected = item["index"] == selected_index
        points = [(point[1], point[0]) for point in item["route_geometry"]]
        color = route_colors[item["station_id"]]
        folium.PolyLine(points, color="#071018", weight=9 if selected else 7, opacity=.35).add_to(map_view)
        folium.PolyLine(points, color=color, weight=6 if selected else 4, opacity=.98 if selected else .85,
                        tooltip=f"{escape(item['station_name'])}: {item['travel_minutes']:.0f} min").add_to(map_view)
    if active_result.get("selection_applied"):
        chosen = active_result["rows"][selected_index]
        chosen_points = [(point[1], point[0]) for point in chosen["route_geometry"]]
        folium.PolyLine(chosen_points, color="#071018", weight=11, opacity=.55).add_to(map_view)
        folium.PolyLine(chosen_points, color="#38bdf8", weight=7, opacity=1,
                        tooltip=f"Selected: {escape(chosen['station_name'])}").add_to(map_view)
    for item in active_result["rows"]:
        recommended = item["index"] == recommended_index
        popup = (f"<b>{escape(item['station_name'])}</b><br>{item['distance_km']:.1f} km | Total {item['total_minutes']:.0f} min"
                 f"<br>Wait {item['wait_minutes']:.0f} min | ₹{item['estimated_cost_inr']:.0f}"
                 f"<br>₹{item['price_inr_kwh']:.2f}/kWh | Queue {item['queue_length']}")
        selected_station = active_result.get("selection_applied") and item["index"] == selected_index
        if selected_station:
            marker = folium.Icon(color="blue", icon="check", prefix="fa")
        elif recommended:
            marker = folium.Icon(color="green", icon="bolt", prefix="fa")
        elif item["station_id"] in routed_ids:
            marker = folium.Icon(color="orange", icon="plug", prefix="fa")
        else:
            marker = folium.Icon(color="lightgray", icon="bolt", prefix="fa")
        folium.Marker([item["latitude"], item["longitude"]], tooltip=escape(item["station_name"]),
                      popup=folium.Popup(popup, max_width=290), icon=marker).add_to(map_view)
    focus = [active_result["location"]] + [[row["latitude"], row["longitude"]] for row in route_items]
    map_view.fit_bounds(focus, padding=(45, 45), max_zoom=14)
    legend_bg, legend_text, legend_line = ("#ffffff", "#14202b", "#dbe3ea") if light_map else ("#101827", "#e5e7eb", "#334155")
    selected_legend = "<br><span style='color:#38bdf8'>━━</span> Selected" if active_result.get("selection_applied") else ""
    legend = f"""<div style='position:fixed;top:14px;right:14px;z-index:9999;background:{legend_bg};color:{legend_text};padding:9px 11px;border-radius:9px;border:1px solid {legend_line};font-size:11px;box-shadow:0 3px 12px rgba(0,0,0,.18)'><b>Routes</b><br><span style='color:#10b981'>━━</span> Recommended<br><span style='color:#f59e0b'>━━</span> Alternative<br><span style='color:#ef5b4f'>━━</span> Other option{selected_legend}</div>"""
    map_view.get_root().html.add_child(Element(legend))
    return map_view


def apply_selection(item: dict) -> None:
    if result.get("selection_applied") and result.get("selected_index") == item["index"]:
        return
    if "selection_base_queue_manager" not in st.session_state:
        st.session_state.selection_base_queue_manager = deepcopy(env.queue_manager)
        st.session_state.selection_base_step = env.step_count
    env.queue_manager = deepcopy(st.session_state.selection_base_queue_manager)
    station_state = env.queue_manager.stations[str(item["station_id"])]
    arrival_status = station_state.arrive(float(item["charge_minutes"]))
    env.step_count = int(st.session_state.selection_base_step) + 1
    result["selected_index"] = item["index"]
    result["selection_applied"] = True
    result["selection_reward"] = item["decision_score"]
    result["transition_info"] = {
        "recommended_station_id": result["rows"][result["recommended_index"]]["station_id"],
        "actual_selected_station_id": item["station_id"],
        "recommendation_accepted": item["index"] == result["recommended_index"],
        "arrival_status": arrival_status,
    }
    st.session_state.recommendation = result
    st.rerun()


control_col, map_col, recommendation_col = st.columns([0.22, 0.50, 0.28], gap="medium")

with control_col:
    st.markdown('<div class="dashboard-panel"><div class="panel-title">Your EV and preferences</div><div class="tagline">Define the trip before comparing stations.</div></div>', unsafe_allow_html=True)
    with st.form("ev_search_form"):
        soc = st.slider("Battery level", 1, 99, 22, format="%d%%")
        target = st.slider("Target charge", soc + 1, 100, 80, format="%d%%")
        capacity = st.number_input("Battery capacity (kWh)", 10.0, 200.0, 50.0, step=1.0)
        connector = st.selectbox("Connector type", ["CCS Type 2", "Type 2", "CHAdeMO", "GB/T", "Other"])
        preference = st.selectbox("Preference", ["Balanced", "Fastest", "Lowest Waiting", "Cheapest"])
        with st.expander("Advanced input"):
            manual_coordinates = st.checkbox("Use manual coordinates")
            manual_latitude = st.number_input("Latitude", -90.0, 90.0, INDIA_CENTER[0], format="%.6f")
            manual_longitude = st.number_input("Longitude", -180.0, 180.0, INDIA_CENTER[1], format="%.6f")
        submitted = st.form_submit_button("FIND CHARGING STATIONS", type="primary", width="stretch")
    if submitted:
        try:
            with st.spinner("Retrieving stations, routes, weather and current estimates..."):
                location = ({"latitude": manual_latitude, "longitude": manual_longitude,
                             "display_name": "Manual coordinates", "source": "manual_input"}
                            if manual_coordinates else cached_geocode(location_query))
                request = EVRequest(location["latitude"], location["longitude"], float(soc), float(capacity),
                                    float(target), connector, preference)
                built = build_recommendation(env, request, location["display_name"])
                built["geocoding_source"] = location["source"]
                st.session_state.pop("selection_base_queue_manager", None)
                st.session_state.pop("selection_base_step", None)
                st.session_state.recommendation = built
                st.rerun()
        except Exception as error:
            st.error(f"Search could not be completed: {error}")
    required = result["required_energy_kwh"] if result else (target - soc) / 100 * capacity
    safe = result["safe_range_km"] if result else soc / 100 * capacity / .18 * .8
    c1, c2 = st.columns(2)
    c1.metric("Required energy", f"{required:.1f} kWh")
    c2.metric("Safe range", f"{safe:.0f} km")
    if result and st.button("Reset scenario", width="stretch"):
        st.session_state.smart_env = initialize_environment()
        st.session_state.recommendation = None
        st.session_state.pop("selection_base_queue_manager", None)
        st.session_state.pop("selection_base_step", None)
        st.rerun()

with map_col:
    if result:
        st.markdown(f'<div class="dashboard-panel"><div class="panel-title">{escape(location_short)}</div>'
                    f'<div class="tagline">Road routes with estimated congestion | {result["pricing_period"]} pricing | {escape(result["preference"])} preference</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="dashboard-panel"><div class="panel-title">Charging network map</div><div class="tagline">Enter a location and run the search to display reachable stations and routes.</div></div>', unsafe_allow_html=True)
    st_folium(create_map(result), use_container_width=True, height=650,
              key=f"planner_map_{result.get('selected_index', 0) if result else 'empty'}_{bool(result and result.get('selection_applied'))}_{env.step_count}")

with recommendation_col:
    if not result:
        st.markdown('<div class="dashboard-panel"><div class="panel-title">Recommendation</div>'
                    '<div class="tagline">The recommendation will appear here after the search.</div>'
                    '<div class="data-note">The comparison includes road travel, charging duration, simulated queue, current weather, peak-hour demand and estimated electricity cost.</div></div>', unsafe_allow_html=True)
    else:
        rows = result["rows"]
        recommended = rows[result["recommended_index"]]
        price_label = "published" if recommended["price_source"] == "openstreetmap_charge_tag" else "estimated"
        st.markdown(f'''<div class="recommendation-card">
          <span class="score-badge">{recommended['decision_score']:.0f} / 100</span>
          <div class="section-label">Recommended station</div>
          <div class="recommendation-title">{escape(recommended['station_name'])}</div>
          <div class="muted">{recommended['travel_minutes']:.0f} min road | {recommended['distance_km']:.1f} km | {recommended['road_congestion_label']} congestion</div>
          <div class="metric-grid">
            <div class="metric-box"><div class="metric-label">Travel time</div><div class="metric-value">{recommended['travel_minutes']:.0f} min</div></div>
            <div class="metric-box"><div class="metric-label">Estimated wait</div><div class="metric-value">{recommended['wait_minutes']:.0f} min</div></div>
            <div class="metric-box"><div class="metric-label">Charging time</div><div class="metric-value">{recommended['charge_minutes']:.0f} min</div></div>
            <div class="metric-box"><div class="metric-label">Total time</div><div class="metric-value total-value">{recommended['total_minutes']:.0f} min</div></div>
            <div class="metric-box"><div class="metric-label">Estimated cost</div><div class="metric-value">₹{recommended['estimated_cost_inr']:.0f}</div></div>
            <div class="metric-box"><div class="metric-label">Simulated availability</div><div class="metric-value">{recommended['free_chargers']} / {recommended['total_chargers']} free</div></div>
          </div>
          <div class="data-note">₹{recommended['price_inr_kwh']:.2f}/kWh ({price_label}) | Queue {recommended['queue_length']} | Peak impact ₹{recommended['peak_cost_impact_inr']:.0f}<br>Connector: {escape(recommended['connector_type'])}{'' if recommended['connector_verified'] else ' — confirm with operator'}</div>
        </div>''', unsafe_allow_html=True)
        recommendation_selected = result.get("selection_applied") and result.get("selected_index") == recommended["index"]
        if recommendation_selected:
            st.button("RECOMMENDATION SELECTED", disabled=True, width="stretch")
        elif st.button("ACCEPT RECOMMENDATION", type="primary", width="stretch"):
            apply_selection(recommended)

        if result.get("selection_applied"):
            selected = rows[result["selected_index"]]
            time_difference = selected["total_minutes"] - recommended["total_minutes"]
            cost_difference = selected["estimated_cost_inr"] - recommended["estimated_cost_inr"]
            distance_difference = selected["distance_km"] - recommended["distance_km"]
            accepted_text = "AI recommendation accepted" if selected["index"] == recommended["index"] else "Alternative station selected"
            st.markdown(f'''<div class="navigation-card"><div class="section-label">Selected route</div>
              <div class="navigation-title">{escape(selected['station_name'])}</div>
              <div class="muted">{accepted_text} | Blue route on the map</div>
              <div class="difference">Compared with recommendation: {time_difference:+.0f} min | {cost_difference:+.0f} ₹ | {distance_difference:+.1f} km</div></div>''', unsafe_allow_html=True)
            directions_query = urlencode({
                "api": 1,
                "origin": f"{result['location'][0]},{result['location'][1]}",
                "destination": f"{selected['latitude']},{selected['longitude']}",
                "travelmode": "driving",
            })
            st.link_button("OPEN DRIVING DIRECTIONS", f"https://www.google.com/maps/dir/?{directions_query}",
                           type="primary", width="stretch")

        st.markdown('<div class="panel-title" style="margin-top:.8rem">Other options</div>', unsafe_allow_html=True)
        alternatives = [item for item in sorted(rows, key=lambda x: (-x["decision_score"], x["total_minutes"]))
                        if item["station_id"] != recommended["station_id"]]
        for item in alternatives[:2]:
            st.markdown(f'''<div class="option-card"><div class="option-name">{escape(item['station_name'])}</div>
              <div class="muted">{item['travel_minutes']:.0f} min road | {item['distance_km']:.1f} km | ₹{item['price_inr_kwh']:.2f}/kWh</div>
              <div class="option-grid"><span><span class="metric-label">Total</span><br><b>{item['total_minutes']:.0f} min</b></span>
              <span><span class="metric-label">Wait</span><br><b>{item['wait_minutes']:.0f} min</b></span>
              <span><span class="metric-label">Cost</span><br><b>₹{item['estimated_cost_inr']:.0f}</b></span></div></div>''', unsafe_allow_html=True)
            item_selected = result.get("selection_applied") and result.get("selected_index") == item["index"]
            if item_selected:
                st.button("STATION SELECTED", key=f"selected_{item['station_id']}", disabled=True, width="stretch")
            elif st.button("SELECT THIS STATION", key=f"select_{item['station_id']}", width="stretch"):
                apply_selection(item)

if not result:
    st.info("Enter any Indian location above, choose the EV details, and select Find Charging Stations.")
    st.stop()

if result.get("selection_applied"):
    chosen = result["rows"][result["selected_index"]]
    st.success(f"Selected station: {chosen['station_name']} | {chosen['distance_km']:.1f} km | {chosen['total_minutes']:.0f} min | ₹{chosen['estimated_cost_inr']:.0f}")

st.caption("Availability and queues are simulator estimates. Tariffs are marked estimated unless a published station tariff is available.")
