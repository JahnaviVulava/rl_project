"""Small callback that saves comparable episode-reward learning curves."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeRewardRecorder(BaseCallback):
    """Record completed episode returns without changing model training."""

    def __init__(self, algorithm: str, output_directory: Path):
        super().__init__()
        self.algorithm = algorithm
        self.output_directory = output_directory
        self.current_return = 0.0
        self.episode_number = 0
        self.rows: list[dict[str, float | int | str]] = []

    def _on_step(self) -> bool:
        self.current_return += float(self.locals["rewards"][0])
        if bool(self.locals["dones"][0]):
            self.episode_number += 1
            self.rows.append({
                "algorithm": self.algorithm,
                "episode": self.episode_number,
                "timesteps": self.num_timesteps,
                "episode_reward": self.current_return,
            })
            self.current_return = 0.0
        return True

    def _on_training_end(self) -> None:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(
            self.output_directory / f"{self.algorithm.lower()}_training_curve.csv",
            index=False,
        )
