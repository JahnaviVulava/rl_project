"""Load a trained policy, falling back transparently to weighted greedy."""
from pathlib import Path
from src.baselines.policies import weighted_greedy_action

def recommend(env,model_path:Path|str|None=None,algorithm="PPO")->tuple[int,str]:
    if model_path and Path(model_path).with_suffix(".zip").exists():
        module=__import__("stable_baselines3",fromlist=[algorithm]); model=getattr(module,algorithm).load(model_path); observation=env._observation(); action,_=model.predict(observation,deterministic=True)
        if env.candidate_mask[int(action)]: return int(action),algorithm.lower()
    return weighted_greedy_action(env),"weighted_greedy_fallback"
