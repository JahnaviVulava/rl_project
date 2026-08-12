"""Reproducible EV request generation near known stations."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EVRequest:
    latitude: float; longitude: float; soc_percent: float; battery_capacity_kwh: float
    target_soc_percent: float; connector_type: str; preference: str = "Balanced"

    @property
    def required_energy_kwh(self) -> float:
        return max(0.0, (self.target_soc_percent-self.soc_percent)/100*self.battery_capacity_kwh)


class EVGenerator:
    def __init__(self, stations: pd.DataFrame, seed: int = 42): self.stations, self.rng = stations, np.random.default_rng(seed)
    def sample(self) -> EVRequest:
        station = self.stations.iloc[int(self.rng.integers(len(self.stations)))]
        soc = float(self.rng.uniform(15, 55)); target = float(self.rng.uniform(max(60, soc+10), 95))
        return EVRequest(float(station.latitude+self.rng.normal(0, .025)), float(station.longitude+self.rng.normal(0, .025)), soc, float(self.rng.choice([40,50,60,75])), target, str(station.connector_type))
