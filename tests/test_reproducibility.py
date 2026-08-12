import numpy as np
from src.environment.ev_charging_env import EVChargingEnv
def test_same_seed_same_trajectory():
    first,second=EVChargingEnv(),EVChargingEnv(); obs1,info1=first.reset(seed=123); obs2,info2=second.reset(seed=123)
    assert np.array_equal(obs1,obs2) and np.array_equal(info1["candidate_mask"],info2["candidate_mask"])
    action=int(np.flatnonzero(info1["candidate_mask"])[0])
    out1=first.step(action); out2=second.step(action)
    assert np.array_equal(out1[0],out2[0]) and out1[1]==out2[1]
