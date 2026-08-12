"""One-command reproducible training and held-out model comparison."""
from __future__ import annotations

import argparse
import pandas as pd

from src.evaluation.compare_models import compare
from src.rl.train_a2c import train as train_a2c
from src.rl.train_dqn import train as train_dqn
from src.rl.train_ppo import train as train_ppo
from src.utils.config import project_path


def run_experiment(timesteps: int = 10000, verbose: int = 0) -> pd.DataFrame:
    """Train PPO, DQN and A2C equally, then evaluate all policies fairly."""
    if timesteps < 1000:
        raise ValueError("Use at least 1,000 timesteps for a meaningful comparison")
    print(f"Training PPO for {timesteps:,} timesteps...")
    train_ppo("review", timesteps, verbose=verbose)
    print(f"Training DQN for {timesteps:,} timesteps...")
    train_dqn("review", timesteps, verbose=verbose)
    print(f"Training A2C for {timesteps:,} timesteps...")
    train_a2c("review", timesteps, verbose=verbose)
    table = compare(save_plots=True)
    print("\nSaved metrics:", project_path("results/metrics/model_comparison.csv"))
    print("Saved plot:", project_path("results/plots/model_comparison.png"))
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--verbose", type=int, default=0, choices=(0, 1))
    arguments = parser.parse_args()
    run_experiment(arguments.timesteps, arguments.verbose)
