# Data flow

```text
Driver input
  → EVRequest (required energy and safe range)
  → OpenStreetMap discovery OR local processed dataset
  → nearby + connector + battery reachability filtering
  → top-K candidates + padding mask
  → OSRM route OR Haversine fallback
  → Open-Meteo weather OR offline weather
  → estimated congestion + stateful queue + price + future demand
  → normalized fixed observation vector
  → PPO/DQN action OR labelled weighted-greedy fallback
  → recommended_station_id
  → driver choice
  → actual_selected_station_id
  → five-minute environment transition at selected station
  → updated occupancy, queue, network balance and next observation
```

Real fields remain distinct from `derived_*`, `estimated_*`, and `simulated_*` fields. The raw dataset remains immutable; preprocessing writes into `data/processed`.
