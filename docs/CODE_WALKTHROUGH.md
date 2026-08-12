# How to understand this project

Read in this execution order:

1. `config/default.yaml` supplies every important assumption: seed, five-minute step, energy use, safety margin, candidate count, price bounds, reward weights and training budgets.
2. `src/data/prepare_data.py` discovers source concepts through aliases, validates coordinates, removes duplicates, standardizes connectors and writes provenance columns. Input is the raw CSV; outputs are `stations.csv` and a JSON report.
3. `src/utils/geo.py` implements Haversine distance, safe range, connector filtering and top-K retrieval.
4. `src/simulation/ev_generator.py` creates reproducible feasible requests. `queue_manager.py` stores remaining session times and FIFO queues. `traffic.py`, `demand.py`, and `pricing.py` provide bounded interpretable estimates.
5. `src/environment/reward.py` exposes seven independent score functions. `ev_charging_env.py` builds the fixed observation, maps action indices to station IDs, advances time, applies the driver's actual station, and returns reward plus a detailed `info` dictionary.
6. `src/baselines/policies.py` implements Random, Nearest, Minimum Queue, Cheapest, Maximum Free, and Weighted Greedy against that same environment.
7. `src/rl/train_ppo.py` and `train_dqn.py` seed libraries, call SB3 `check_env`, perform random rollouts, train, log to TensorBoard, and save models.
8. `src/rl/inference.py` loads PPO when present and openly falls back to weighted greedy when not.
9. `pages/1_Smart_Recommendation.py` converts driver inputs into `EVRequest`, builds candidates, requests an action, displays computed values, then calls `set_actual_selection`. The subsequent `step` updates the chosen station only.

Important input/output examples:

```python
request = EVRequest(10.01, 76.31, 20, 50, 80, "CCS Type 2")
assert request.required_energy_kwh == 30

observation, info = env.reset(seed=42)  # float array and candidate mask
env.set_actual_selection(2)             # driver rejects recommendation 0
observation, reward, done, truncated, info = env.step(0)
assert info["actual_selected_station_id"] != info["recommended_station_id"]
```
