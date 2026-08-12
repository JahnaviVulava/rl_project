"""Train DQN under the same environment, seeds and budget as PPO."""
from src.environment.ev_charging_env import EVChargingEnv
from src.rl.common import set_global_seeds,validate_environment
from src.rl.training_callback import EpisodeRewardRecorder
from src.utils.config import load_config,project_path

def train(mode="review",timesteps=None,verbose=1):
    from stable_baselines3 import DQN
    config=load_config(mode); seed=config["seed"]; set_global_seeds(seed); env=EVChargingEnv(config=config); validate_environment(env)
    steps=int(timesteps or config["training"]["dqn_timesteps"]); callback=EpisodeRewardRecorder("DQN",project_path("results/metrics")); model=DQN("MlpPolicy",env,seed=seed,verbose=verbose,tensorboard_log=str(project_path("results/logs/dqn")),learning_starts=min(500,steps//10),buffer_size=max(10000,steps),train_freq=4,target_update_interval=500)
    model.learn(total_timesteps=steps,callback=callback,progress_bar=False); path=project_path("models/dqn_ev_charging"); model.save(path); return model
if __name__=="__main__": train()
