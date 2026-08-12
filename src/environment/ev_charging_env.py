"""Gymnasium MDP for sequential charging-station recommendations.

State: 8 normalized EV/context values plus 11 values per candidate.
Action: one index in ``Discrete(K)``. Padded actions are rejected safely.
Transition: time advances, then the *actual selected station* receives the EV.
Reward: weighted driver convenience and network quality in [0, 100].
"""
from __future__ import annotations
import logging
import math
from typing import Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from src.environment.reward import calculate_reward
from src.simulation.demand import estimate_future_demand
from src.simulation.ev_generator import EVGenerator, EVRequest
from src.simulation.pricing import estimate_price
from src.simulation.queue_manager import QueueManager
from src.simulation.traffic import estimate_traffic
from src.utils.config import load_config, project_path
from src.utils.geo import filter_candidates, safe_range_km

LOGGER = logging.getLogger(__name__)
GLOBAL_FEATURES, CANDIDATE_FEATURES = 8, 11


class EVChargingEnv(gym.Env):
    """A reproducible, laptop-sized EV charging network environment."""
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(self, stations: pd.DataFrame | None = None, config: dict | None = None, render_mode: str | None = None, debug: bool = False):
        super().__init__()
        self.config = config or load_config("review")
        self.k = int(self.config["candidate_count"])
        if stations is None:
            path = project_path(self.config["paths"]["processed_data"])
            if not path.exists():
                from src.data.prepare_data import prepare_data
                stations, _ = prepare_data(config=self.config)
            else: stations = pd.read_csv(path)
        if len(stations) < 2: raise ValueError("At least two stations are required")
        self.stations = stations.reset_index(drop=True).copy()
        if self.config.get("mode") == "review":
            # A dense city/connector cluster makes sequential congestion observable
            # within 50–100 review requests instead of spreading EVs across India.
            grouping = self.stations.groupby(["city", "connector_type"], dropna=False).size()
            city, connector = grouping.idxmax()
            cluster = self.stations[(self.stations.city == city) & (self.stations.connector_type == connector)]
            limit = int(self.config["network"]["review_station_count"])
            self.stations = cluster.head(limit).reset_index(drop=True).copy()
        self.action_space = spaces.Discrete(self.k)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(GLOBAL_FEATURES+self.k*CANDIDATE_FEATURES,), dtype=np.float32)
        self.render_mode, self.debug = render_mode, debug
        self._seed = int(self.config["seed"])
        self.step_count = 0
        self.request: EVRequest | None = None
        self.candidates = pd.DataFrame()
        self.candidate_mask = np.zeros(self.k, dtype=bool)
        self.pending_actual_selection: int | None = None
        self.queue_manager = self._new_queue_manager()

    def _new_queue_manager(self) -> QueueManager:
        counts = self.stations.number_of_chargers.astype(int).copy()
        if self.config.get("mode") == "review" and "number_of_chargers_source" in self.stations:
            assumed = self.stations.number_of_chargers_source.eq("simulated_default")
            counts.loc[assumed] = int(self.config["simulation"]["default_chargers"])
        return QueueManager(dict(zip(self.stations.station_id.astype(str), counts)))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        used_seed = self._seed if seed is None else seed
        self.generator = EVGenerator(self.stations, used_seed)
        self.queue_manager = self._new_queue_manager()
        self.step_count = 0
        self.hour = int(self.np_random.integers(0, 24)); self.day_of_week = int(self.np_random.integers(0, 7))
        self.weather_severity = float(self.np_random.uniform(0, .8)); self.temperature_c = float(self.np_random.uniform(18, 38)); self.rain_mm = 8*self.weather_severity
        self._new_request()
        observation = self._observation()
        return observation, {"candidate_mask": self.candidate_mask.copy(), "station_ids": self.candidates.station_id.tolist()}

    def _new_request(self) -> None:
        self.request = self.generator.sample()
        simulation = self.config["simulation"]
        reach = safe_range_km(self.request.soc_percent, self.request.battery_capacity_kwh, simulation["consumption_kwh_km"], simulation["safe_range_factor"])
        self.candidates = filter_candidates(self.stations, self.request.latitude, self.request.longitude, self.request.connector_type, reach, self.k)
        # Data can be sparse for a connector; ensure at least the generating station remains possible.
        self.candidate_mask = np.zeros(self.k, dtype=bool); self.candidate_mask[:len(self.candidates)] = True
        self.safe_range = reach
        self.traffic = estimate_traffic(self.hour, self.day_of_week, self.weather_severity, self.np_random)

    def _candidate_metrics(self, row: pd.Series) -> dict[str, Any]:
        state = self.queue_manager.stations[str(row.station_id)]
        road_distance = row.get("road_distance_km") if hasattr(row, "get") else None
        route_duration = row.get("base_route_duration_minutes") if hasattr(row, "get") else None
        if pd.notna(road_distance) and pd.notna(route_duration):
            distance = float(road_distance)
            base_minutes = float(route_duration)
        else:
            distance = float(row.distance_km)*1.25  # DERIVED: offline road-length approximation.
            base_minutes = distance/self.config["simulation"]["estimated_speed_kmh"]*60
        # Offline candidates and candidates regenerated after env.step() do not
        # carry a live route field. Series.get avoids pandas raising KeyError.
        route_congestion = row.get("road_congestion_score", self.traffic) if hasattr(row, "get") else self.traffic
        road_congestion_score = float(self.traffic if pd.isna(route_congestion) else route_congestion)
        travel = base_minutes*(1+0.75*road_congestion_score)
        effective_power = max(1.0, float(row.power_kw)*self.config["simulation"]["charging_efficiency"]*(1-0.04*self.weather_severity))
        charge = self.request.required_energy_kwh/effective_power*60
        demand = estimate_future_demand(self.hour, state.queue_length, self.traffic, self.weather_severity, state.utilization)
        price = estimate_price(float(row.base_price_inr_kwh), self.hour, state.utilization, demand, self.config["pricing"])
        return {"distance_km":distance,"base_travel_minutes":base_minutes,"road_congestion_score":road_congestion_score,"travel_minutes":travel,"wait_minutes":state.estimate_wait_minutes(charge),"charge_minutes":charge,"price_inr_kwh":price,"estimated_cost_inr":price*self.request.required_energy_kwh,"future_demand":demand,"utilization":state.utilization,"free_ratio":state.free_chargers/state.total_chargers,"queue_length":state.queue_length,"compatible":True,"reachable":distance<=self.safe_range,"valid":True}

    def _observation(self) -> np.ndarray:
        global_values = [self.request.soc_percent/100, min(self.request.required_energy_kwh/100,1), min(self.safe_range/500,1), self.hour/23, self.weather_severity, np.clip((self.temperature_c+10)/60,0,1), min(self.rain_mm/50,1), self.traffic]
        candidate_values: list[float] = []
        for index in range(self.k):
            if index >= len(self.candidates): candidate_values.extend([0.0]*CANDIDATE_FEATURES); continue
            row=self.candidates.iloc[index]; m=self._candidate_metrics(row)
            candidate_values.extend([min(m["distance_km"]/100,1),min(m["travel_minutes"]/180,1),min(m["queue_length"]/20,1),min(m["wait_minutes"]/180,1),m["free_ratio"],m["utilization"],min(float(row.power_kw)/350,1),min(m["price_inr_kwh"]/40,1),m["future_demand"],1.0,float(m["reachable"])])
        observation=np.asarray(global_values+candidate_values,dtype=np.float32)
        assert observation.shape==self.observation_space.shape and np.isfinite(observation).all()
        return np.clip(observation,0,1)

    def set_actual_selection(self, candidate_index: int) -> None:
        """Set the driver's choice for the next step; recommendation remains unchanged."""
        self.pending_actual_selection = int(candidate_index)

    def step(self, action: int):
        recommended_index=int(action); actual_index=recommended_index if self.pending_actual_selection is None else self.pending_actual_selection
        self.pending_actual_selection=None
        self.queue_manager.advance_all(self.config["simulation"]["step_minutes"])
        failure_reasons=[]
        if not self.action_space.contains(recommended_index) or not self.candidate_mask[recommended_index]: failure_reasons.append("invalid_candidate")
        if not self.action_space.contains(actual_index) or not self.candidate_mask[actual_index]: failure_reasons.append("invalid_actual_selection")
        if failure_reasons:
            reward=0.0; components={}; metrics={"travel_minutes":0.0,"wait_minutes":0.0,"charge_minutes":0.0,"estimated_cost_inr":0.0}
            recommended_id=actual_id=None
        else:
            recommended_row=self.candidates.iloc[recommended_index]; actual_row=self.candidates.iloc[actual_index]
            recommended_id=str(recommended_row.station_id); actual_id=str(actual_row.station_id)
            metrics=self._candidate_metrics(actual_row)
            if not metrics["reachable"]: failure_reasons.append("unreachable_station")
            before=self.queue_manager.snapshot()[actual_id]
            self.queue_manager.stations[actual_id].arrive(metrics["charge_minutes"])
            utilizations=[state.utilization for state in self.queue_manager.stations.values()]
            metrics["utilization_std"]=float(np.std(utilizations)); metrics["valid"]=not failure_reasons
            reward,components=calculate_reward(metrics,self.config["reward"])
            metrics["before_state"],metrics["after_state"]=before,self.queue_manager.snapshot()[actual_id]
        self.step_count+=1
        terminated=self.step_count>=int(self.config["simulation"]["episode_steps"])
        self.hour=(self.hour+self.config["simulation"]["step_minutes"]//60)%24
        network_states=list(self.queue_manager.stations.values())
        network_utilizations=[state.utilization for state in network_states]
        network_queues=[state.queue_length for state in network_states]
        info={"recommended_station_id":recommended_id,"actual_selected_station_id":actual_id,"recommendation_accepted":recommended_index==actual_index,"failure_reasons":failure_reasons,"reward_components":components,"average_queue":float(np.mean(network_queues)),"maximum_queue":int(max(network_queues)),"mean_utilization":float(np.mean(network_utilizations)),"maximum_utilization":float(max(network_utilizations)),"utilization_imbalance":float(np.std(network_utilizations)),**metrics}
        if not terminated: self._new_request()
        observation=self._observation()
        if self.debug: self._debug_print(recommended_index,actual_index,reward,info)
        if self.render_mode=="human": print(self.render())
        return observation,float(reward),terminated,False,info

    def _debug_print(self,recommended:int,actual:int,reward:float,info:dict)->None:
        print(f"STEP {self.step_count}\nEV SOC: {self.request.soc_percent:.1f}%  Required Energy: {self.request.required_energy_kwh:.1f} kWh  Safe Range: {self.safe_range:.1f} km\nENVIRONMENT Weather severity: {self.weather_severity:.2f}  Traffic: {self.traffic:.2f}  Hour: {self.hour}\nACTION Recommended index: {recommended}  Actual index: {actual}\nRESULT Travel: {info.get('travel_minutes',0):.1f}  Wait: {info.get('wait_minutes',0):.1f}  Charge: {info.get('charge_minutes',0):.1f}  Cost: {info.get('estimated_cost_inr',0):.1f}\nREWARD {reward:.2f} {info.get('reward_components',{})}")

    def render(self):
        snap=self.queue_manager.snapshot(); occupied=sum(x["occupied"] for x in snap.values()); waiting=sum(x["queue"] for x in snap.values())
        return f"Step {self.step_count}: occupied={occupied}, waiting={waiting}, candidates={int(self.candidate_mask.sum())}"

    def close(self): pass
