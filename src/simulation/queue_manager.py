"""Discrete-time charger and FIFO queue simulation.

Each active session stores remaining minutes. Advancing time releases completed
chargers and immediately promotes waiting EVs, so queues cannot remain frozen.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math


@dataclass
class StationQueue:
    total_chargers: int
    active_sessions: list[float] = field(default_factory=list)
    waiting_durations: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.total_chargers < 1: raise ValueError("A station needs at least one charger")

    @property
    def occupied_chargers(self) -> int: return len(self.active_sessions)
    @property
    def free_chargers(self) -> int: return self.total_chargers - self.occupied_chargers
    @property
    def queue_length(self) -> int: return len(self.waiting_durations)
    @property
    def utilization(self) -> float: return self.occupied_chargers / self.total_chargers

    def arrive(self, charging_minutes: float) -> str:
        """Start charging or join FIFO queue; returns ``charging`` or ``queued``."""
        if not math.isfinite(charging_minutes) or charging_minutes <= 0: raise ValueError("Charging duration must be positive")
        if self.free_chargers:
            self.active_sessions.append(float(charging_minutes)); return "charging"
        self.waiting_durations.append(float(charging_minutes)); return "queued"

    def estimate_wait_minutes(self, new_duration: float = 0.0) -> float:
        """Simulate charger availability without mutating the actual state."""
        if self.free_chargers: return 0.0
        availability = sorted(self.active_sessions)
        for duration in self.waiting_durations:
            earliest = availability.pop(0)
            availability.append(earliest + duration)
            availability.sort()
        return max(0.0, availability[0])

    def advance(self, minutes: float) -> dict[str, int]:
        """Advance all sessions, release completions and promote queued EVs."""
        if minutes <= 0: raise ValueError("Advance duration must be positive")
        completed = 0
        self.active_sessions = [remaining - minutes for remaining in self.active_sessions]
        completed += sum(remaining <= 1e-9 for remaining in self.active_sessions)
        self.active_sessions = [remaining for remaining in self.active_sessions if remaining > 1e-9]
        promoted = 0
        while self.waiting_durations and self.free_chargers:
            self.active_sessions.append(self.waiting_durations.pop(0)); promoted += 1
        assert 0 <= self.occupied_chargers <= self.total_chargers
        return {"completed": completed, "promoted": promoted}


class QueueManager:
    """Own the mutable queue state for all stations."""
    def __init__(self, station_chargers: dict[str, int]):
        self.stations = {station_id: StationQueue(int(count)) for station_id, count in station_chargers.items()}

    def advance_all(self, minutes: float) -> None:
        for station in self.stations.values(): station.advance(minutes)

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {station_id: {"occupied": state.occupied_chargers, "free": state.free_chargers, "queue": state.queue_length, "utilization": state.utilization, "estimated_wait_minutes": state.estimate_wait_minutes()} for station_id, state in self.stations.items()}
