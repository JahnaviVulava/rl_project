"""Episode-level metric aggregation."""
import numpy as np

def summarize(records:list[dict])->dict:
    if not records: return {}
    keys=["reward","travel_minutes","wait_minutes","charge_minutes","estimated_cost_inr","average_queue","maximum_queue","mean_utilization","maximum_utilization","utilization_imbalance"]
    result={f"mean_{key}":float(np.mean([r.get(key,0) for r in records])) for key in keys}
    result.update({"failure_rate":float(np.mean([bool(r.get("failure_reasons")) for r in records])),"mean_reward_std":float(np.std([r.get("reward",0) for r in records]))})
    return result
