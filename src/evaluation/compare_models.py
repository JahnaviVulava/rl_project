"""Fair comparison of baselines and saved RL models on held-out seeds."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.baselines import policies
from src.environment.ev_charging_env import EVChargingEnv
from src.utils.config import load_config, project_path

Policy = Callable[[EVChargingEnv], int]
EVALUATION_SEEDS = (101, 102, 103, 104, 105)  # Separate from training seed 42.
BASELINES: dict[str, Policy] = {
    "Random": policies.random_action,
    "Nearest": policies.nearest_action,
    "Minimum Queue": policies.min_queue_action,
    "Cheapest": policies.cheapest_action,
    "Maximum Free": policies.maximum_free_action,
    "Weighted Greedy": policies.weighted_greedy_action,
}


def _episode_metrics(policy: Policy, seed: int, steps: int) -> dict[str, float]:
    """Evaluate one complete held-out scenario and return its mean metrics."""
    env = EVChargingEnv(config=load_config("review"))
    env.reset(seed=seed)
    rows: list[dict] = []
    for _ in range(steps):
        action = policy(env)
        _, reward, terminated, truncated, info = env.step(action)
        rows.append({"reward": reward, **info})
        if terminated or truncated:
            break
    frame = pd.DataFrame(rows)
    total_time = frame["travel_minutes"] + frame["wait_minutes"] + frame["charge_minutes"]
    return {
        "reward": float(frame.reward.mean()),
        "travel_minutes": float(frame.travel_minutes.mean()),
        "wait_minutes": float(frame.wait_minutes.mean()),
        "charge_minutes": float(frame.charge_minutes.mean()),
        "total_time_minutes": float(total_time.mean()),
        "estimated_cost_inr": float(frame.estimated_cost_inr.mean()),
        "failure_rate": float(frame.failure_reasons.map(bool).mean()),
        "average_queue": float(frame.average_queue.mean()),
        "maximum_queue": float(frame.maximum_queue.max()),
        "mean_utilization": float(frame.mean_utilization.mean()),
        "utilization_imbalance": float(frame.utilization_imbalance.mean()),
    }


def evaluate_policy(name: str, policy: Policy, seeds=EVALUATION_SEEDS, steps: int = 75) -> dict:
    """Report mean ± between-scenario standard deviation across held-out seeds."""
    episodes = pd.DataFrame([_episode_metrics(policy, seed, steps) for seed in seeds])
    result: dict[str, float | str] = {"model": name}
    for column in episodes.columns:
        result[f"mean_{column}"] = float(episodes[column].mean())
        result[f"std_{column}"] = float(episodes[column].std(ddof=0))
    result["success_rate"] = 1.0 - float(result["mean_failure_rate"])
    return result


def _load_rl_policies() -> dict[str, Policy]:
    """Load only models that truly exist; absent models never get fake rows."""
    result: dict[str, Policy] = {}
    for algorithm in ("PPO", "DQN", "A2C"):
        model_file = project_path(f"models/{algorithm.lower()}_ev_charging.zip")
        if not model_file.exists():
            continue
        model_class = getattr(__import__("stable_baselines3", fromlist=[algorithm]), algorithm)
        model = model_class.load(model_file)

        def rl_policy(env: EVChargingEnv, loaded=model) -> int:
            action, _ = loaded.predict(env._observation(), deterministic=True)
            index = int(action)
            return index if env.candidate_mask[index] else int(np.flatnonzero(env.candidate_mask)[0])

        result[algorithm] = rl_policy
    return result


def _model_verdict(row: pd.Series, random_reward: float) -> str:
    """Use explicit thresholds instead of making unsupported quality claims."""
    if row.mean_failure_rate > 0.05:
        return "Needs improvement: failure rate exceeds 5%."
    if row.mean_reward >= 75 and row.mean_reward >= random_reward + 2:
        return "Good: reward is at least 75 and meaningfully beats Random."
    if row.mean_reward >= 75:
        return "Acceptable: strong absolute reward, but no clear advantage over Random."
    return "Needs improvement: mean reward is below 75."


def compare(save_plots: bool = True) -> pd.DataFrame:
    """Evaluate every available policy and persist results, plots and verdicts."""
    evaluated = {**BASELINES, **_load_rl_policies()}
    table = pd.DataFrame([evaluate_policy(name, policy) for name, policy in evaluated.items()])
    table = table.sort_values("mean_reward", ascending=False).reset_index(drop=True)
    random_reward = float(table.loc[table.model == "Random", "mean_reward"].iloc[0])
    table["verdict"] = table.apply(lambda row: _model_verdict(row, random_reward), axis=1)

    metrics_directory = project_path("results/metrics")
    plots_directory = project_path("results/plots")
    metrics_directory.mkdir(parents=True, exist_ok=True)
    plots_directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(metrics_directory / "model_comparison.csv", index=False)

    rl_rows = table[table.model.isin(["PPO", "DQN", "A2C"])]
    best_rl_models: list[str] = []
    if not rl_rows.empty:
        best_rl_reward = float(rl_rows.mean_reward.max())
        best_rl_models = rl_rows.loc[
            np.isclose(rl_rows.mean_reward, best_rl_reward, atol=1e-9), "model"
        ].tolist()
    summary = {
        "best_evaluated_model": str(table.iloc[0].model),
        "best_rl_models": best_rl_models,
        "training_seed": 42,
        "held_out_evaluation_seeds": list(EVALUATION_SEEDS),
        "selection_basis": "highest mean reward across held-out scenarios",
        "accuracy_note": "RL has no classification accuracy; use success rate and evaluation metrics.",
    }
    (metrics_directory / "model_selection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if save_plots:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].bar(table.model, table.mean_reward, yerr=table.std_reward, capsize=3)
        axes[0].set_ylabel("Mean reward (0–100)")
        axes[0].set_title("Held-out reward: mean ± standard deviation")
        axes[0].tick_params(axis="x", rotation=45)
        axes[1].bar(table.model, table.mean_total_time_minutes)
        axes[1].set_ylabel("Minutes")
        axes[1].set_title("Mean total expected time")
        axes[1].tick_params(axis="x", rotation=45)
        figure.tight_layout()
        figure.savefig(plots_directory / "model_comparison.png", dpi=160)
        plt.close(figure)

    display_columns = ["model", "mean_reward", "std_reward", "success_rate",
                       "mean_total_time_minutes", "mean_wait_minutes",
                       "mean_estimated_cost_inr", "mean_utilization_imbalance", "verdict"]
    print(table[display_columns].to_string(index=False))
    print(summary)
    return table


if __name__ == "__main__":
    compare()
