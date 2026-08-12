"""Train PPO after mandatory Gymnasium validation."""
from src.environment.ev_charging_env import EVChargingEnv
from src.rl.common import set_global_seeds,validate_environment
from src.rl.training_callback import EpisodeRewardRecorder
from src.utils.config import load_config,project_path

def train(mode="review",timesteps=None,verbose=1):
    from stable_baselines3 import PPO
    config=load_config(mode); seed=config["seed"]; set_global_seeds(seed); env=EVChargingEnv(config=config); validate_environment(env)
    steps=int(timesteps or config["training"]["ppo_timesteps"]); callback=EpisodeRewardRecorder("PPO",project_path("results/metrics")); model=PPO("MlpPolicy",env,seed=seed,verbose=verbose,n_steps=128,batch_size=64,tensorboard_log=str(project_path("results/logs/ppo")))
    model.learn(total_timesteps=steps,callback=callback,progress_bar=False); path=project_path("models/ppo_ev_charging"); model.save(path); return model
if __name__=="__main__": train()
