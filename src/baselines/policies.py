"""Conventional policies sharing the same candidate environment."""
import numpy as np

def valid_indices(env): return np.flatnonzero(env.candidate_mask)
def random_action(env): return int(env.np_random.choice(valid_indices(env)))
def nearest_action(env): return int(env.candidates.distance_km.to_numpy().argmin())
def min_queue_action(env): return min(valid_indices(env), key=lambda i: env.queue_manager.stations[str(env.candidates.iloc[i].station_id)].queue_length)
def cheapest_action(env): return min(valid_indices(env), key=lambda i: env._candidate_metrics(env.candidates.iloc[i])["price_inr_kwh"])
def maximum_free_action(env): return max(valid_indices(env), key=lambda i: env.queue_manager.stations[str(env.candidates.iloc[i].station_id)].free_chargers)
def weighted_greedy_action(env):
    return min(valid_indices(env), key=lambda i: (lambda m: .4*m["travel_minutes"]+.4*m["wait_minutes"]+.2*m["price_inr_kwh"])(env._candidate_metrics(env.candidates.iloc[i])))
