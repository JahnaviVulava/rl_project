import numpy as np
from src.environment.ev_charging_env import EVChargingEnv
def test_shapes_actions_and_metrics():
    env=EVChargingEnv(); observation,info=env.reset(seed=42)
    assert observation.shape==env.observation_space.shape and np.isfinite(observation).all()
    action=int(np.flatnonzero(info["candidate_mask"])[0]); _,reward,_,_,step_info=env.step(action)
    assert env.action_space.contains(action) and 0<=reward<=100
    assert step_info["travel_minutes"]>=0 and step_info["wait_minutes"]>=0 and step_info["charge_minutes"]>=0
def test_driver_choice_updates_actual_not_recommended():
    env=EVChargingEnv(); _,info=env.reset(seed=42); valid=np.flatnonzero(info["candidate_mask"])
    if len(valid)<2: return
    recommended,actual=map(int,valid[:2]); recommended_id=env.candidates.iloc[recommended].station_id; actual_id=env.candidates.iloc[actual].station_id
    before_recommended=env.queue_manager.stations[recommended_id].occupied_chargers
    env.set_actual_selection(actual); _,_,_,_,result=env.step(recommended)
    assert result["recommended_station_id"]==recommended_id and result["actual_selected_station_id"]==actual_id and not result["recommendation_accepted"]
    assert env.queue_manager.stations[recommended_id].occupied_chargers==before_recommended
