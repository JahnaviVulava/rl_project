# SmartCharge RL

SmartCharge RL recommends an EV charging station while accounting for how each assignment changes future queues and network congestion. It is a reproducible seventh-semester reinforcement-learning project with a Gymnasium environment, PPO/DQN training, conventional baselines, tests, notebooks, and a Streamlit website.

## Why reinforcement learning?

Nearest-station selection optimizes one request. A station assignment changes occupancy, queues, later waiting time, and therefore later recommendations. This delayed effect makes the task a sequential Markov decision process (MDP). The AI recommends only: `recommended_station_id` and `actual_selected_station_id` are separate, and only the driver's selection changes the network.

## Data and provenance

The supplied `data/raw/Indian_EV_Stations_Simplified.csv` is the [EV Charging Stations in India Simplified 2025 Kaggle dataset](https://www.kaggle.com/datasets/pranjal9091/ev-charging-stations-in-india-simplified-2025). It has 855 rows with real names, cities, states, coordinates, operators, connector types and power. Charger counts and current tariffs are absent, so preprocessing labels their configurable values `simulated_default`; missing power uses `derived_median`. The raw CSV is never overwritten.

Live mode can use OpenStreetMap/Overpass stations, OSRM road routes, and Open-Meteo weather. All have offline fallbacks. OSRM base duration is adjusted by an **estimated congestion** model; it is never called live traffic. Queue, future demand, and prices are clearly labelled simulated or estimated.

## Architecture and MDP

`prepare_data` cleans and records provenance. Candidate retrieval filters nearby stations by connector and safe battery range before taking top K. The queue manager owns active sessions with remaining durations plus FIFO wait lists. Each five-minute Gym step releases finished sessions and starts queued EVs.

The state has 8 normalized EV/context values plus 11 values for each of K stations: route distance/time, queue/wait, free ratio/utilization, power, estimated tariff/demand, compatibility, and reachability. The action is `Discrete(K)`. Padded candidates have a mask and invalid actions fail safely. The 0–100 reward combines waiting (25%), travel (20%), cost (15%), utilization (10%), balance (10%), future demand (10%), and compatibility (10%). Weights live in YAML.

PPO uses a clipped policy-gradient update; DQN learns action values with replay and a target network. Both use MLPs, which are neural-network function approximators—not RL algorithms. They receive identical environments, budgets, seeds and evaluation metrics. No model is hard-coded as the winner.

## Windows PowerShell setup

```powershell
cd C:\Users\jahav\OneDrive\janudrive\project\EV-Charging-RL-Optimizer
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python -m src.data.prepare_data
pytest -v
python scripts/check_environment.py
```

Train and compare:

```powershell
python -m src.rl.train_ppo
python -m src.rl.train_dqn
python -m src.rl.train_a2c
python -m src.evaluation.compare_models
tensorboard --logdir results\logs
```

To reproduce the complete equal-budget experiment in one command:

```powershell
python -m src.evaluation.train_and_compare --timesteps 10000
```

Notebook `04_rl_model_comparison.ipynb` calls this same Python pipeline. Set
`RUN_TRAINING = True` to retrain inside VS Code, or leave it `False` to load the
saved models and reproduce their held-out evaluation and plots quickly.

Run the website:

```powershell
streamlit run app.py
```

Open notebooks directly in VS Code and select the project's `venv` kernel.

## Project tour

- `config/`: reproducible offline, review and live settings.
- `src/data/`: schema-aware cleaning and provenance report.
- `src/simulation/`: EV generation, queues, traffic, demand and pricing.
- `src/environment/`: Gymnasium MDP and independently tested reward scores.
- `src/providers/`: timeout-protected free APIs and offline fallbacks.
- `src/baselines/`, `src/rl/`, `src/evaluation/`: fair policy training/evaluation.
- `pages/`: recommendation, monitor, analysis, simulation and methodology.
- `notebooks/`: seven executable teaching/review workflows.
- `tests/`: invariants for data, distance, queues, candidates, reward and reproducibility.

See [CODE_WALKTHROUGH.md](docs/CODE_WALKTHROUGH.md), [DATA_FLOW.md](docs/DATA_FLOW.md), and [MODEL_EXPLANATION.md](docs/MODEL_EXPLANATION.md).

## Interpreting output

Scores are: 90–100 Excellent, 75–89 Very Good, 60–74 Good, 40–59 Average, 20–39 Poor, below 20 Failed/Very Poor. Total expected time equals travel + waiting + charging. Network imbalance is the standard deviation of utilization: high values mean uneven loading.

## Limitations and future work

Public data does not provide real-time queues or universal tariffs. Offline routes approximate road distance; live OSRM has no Google-style traffic. Standard SB3 PPO has safe invalid-action rejection rather than gradient-level action masking. Evaluation quality depends on training budget; review settings intentionally run quickly. Future work can add authenticated station availability, learned demand from temporal history, calibrated local traffic, MaskablePPO, richer driver acceptance behavior, and city/state/national hierarchical retrieval.
