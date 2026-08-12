import pandas as pd
from src.utils.geo import filter_candidates
def test_unreachable_and_incompatible_are_rejected():
    stations=pd.DataFrame([{"station_id":"near-good","latitude":0,"longitude":.01,"connector_type":"CCS Type 2"},{"station_id":"far","latitude":10,"longitude":10,"connector_type":"CCS Type 2"},{"station_id":"wrong","latitude":0,"longitude":.01,"connector_type":"CHAdeMO"}])
    result=filter_candidates(stations,0,0,"CCS Type 2",10,10)
    assert result.station_id.tolist()==["near-good"]
def test_invalid_padded_action_is_rejected():
    from src.environment.ev_charging_env import EVChargingEnv
    env=EVChargingEnv(); env.reset(seed=42); padded=[i for i,valid in enumerate(env.candidate_mask) if not valid]
    if padded:
        _,reward,_,_,info=env.step(padded[0]); assert reward==0 and "invalid_candidate" in info["failure_reasons"]
