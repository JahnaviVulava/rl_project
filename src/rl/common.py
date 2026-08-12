"""Shared validation and reproducibility helpers for SB3 training."""
import random
import numpy as np

def set_global_seeds(seed:int)->None:
    random.seed(seed); np.random.seed(seed)
    try:
        import torch; torch.manual_seed(seed)
    except ImportError: pass

def validate_environment(env)->None:
    from stable_baselines3.common.env_checker import check_env
    check_env(env,warn=True)
    observation,_=env.reset(seed=env.config["seed"])
    for _ in range(30):
        valid=np.flatnonzero(env.candidate_mask); action=int(env.np_random.choice(valid)); observation,reward,terminated,truncated,_=env.step(action)
        assert np.isfinite(observation).all() and np.isfinite(reward)
        if terminated or truncated: observation,_=env.reset()
    print("Environment validation passed.")
