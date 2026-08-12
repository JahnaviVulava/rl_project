"""Train A2C under the same environment and budget as PPO and DQN."""
from src.environment.ev_charging_env import EVChargingEnv
from src.rl.common import set_global_seeds, validate_environment
from src.rl.training_callback import EpisodeRewardRecorder
from src.utils.config import load_config, project_path


def train(mode="review", timesteps=None, verbose=1):
    from stable_baselines3 import A2C

    config = load_config(mode)
    seed = int(config["seed"])
    set_global_seeds(seed)
    env = EVChargingEnv(config=config)
    validate_environment(env)
    steps = int(timesteps or config["training"]["a2c_timesteps"])
    callback = EpisodeRewardRecorder("A2C", project_path("results/metrics"))
    model = A2C("MlpPolicy", env, seed=seed, verbose=verbose, n_steps=20,
                tensorboard_log=str(project_path("results/logs/a2c")))
    model.learn(total_timesteps=steps, callback=callback, progress_bar=False)
    model.save(project_path("models/a2c_ev_charging"))
    return model


if __name__ == "__main__":
    train()
