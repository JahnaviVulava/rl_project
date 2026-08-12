"""Independent, testable reward score functions producing a 0–100 score."""
from __future__ import annotations
import numpy as np

def _decreasing(value: float, bad_at: float) -> float: return float(100*np.clip(1-value/bad_at, 0, 1))
def calculate_travel_score(minutes: float) -> float: return _decreasing(minutes, 120)
def calculate_wait_score(minutes: float) -> float: return _decreasing(minutes, 180)
def calculate_cost_score(price_inr_kwh: float) -> float: return _decreasing(max(0, price_inr_kwh-8), 32)
def calculate_utilization_score(utilization: float) -> float: return float(100*np.clip(1-utilization, 0, 1))
def calculate_balance_score(std_utilization: float) -> float: return float(100*np.clip(1-std_utilization, 0, 1))
def calculate_demand_score(future_demand: float) -> float: return float(100*np.clip(1-future_demand, 0, 1))
def calculate_compatibility_score(compatible: bool, reachable: bool = True) -> float: return 100.0 if compatible and reachable else 0.0

def calculate_reward(metrics: dict, weights: dict) -> tuple[float, dict[str,float]]:
    """Return weighted reward and component scores; invalid actions score zero."""
    components = {"waiting": calculate_wait_score(metrics["wait_minutes"]), "travel": calculate_travel_score(metrics["travel_minutes"]), "cost": calculate_cost_score(metrics["price_inr_kwh"]), "utilization": calculate_utilization_score(metrics["utilization"]), "balance": calculate_balance_score(metrics["utilization_std"]), "demand": calculate_demand_score(metrics["future_demand"]), "compatibility": calculate_compatibility_score(metrics["compatible"], metrics["reachable"])}
    if not metrics.get("valid", True) or not metrics["compatible"] or not metrics["reachable"]: return 0.0, components
    if abs(sum(weights.values())-1.0) > 1e-8: raise ValueError("Reward weights must sum to 1")
    reward = sum(weights[name]*score for name,score in components.items())
    return float(np.clip(reward,0,100)), components

def interpret_reward(reward: float) -> str:
    if reward >= 90: return "Excellent"
    if reward >= 75: return "Very Good"
    if reward >= 60: return "Good"
    if reward >= 40: return "Average"
    if reward >= 20: return "Poor"
    return "Failed / Very Poor"
