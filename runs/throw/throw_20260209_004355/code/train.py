"""
Residual RL Training Script for: throw
Task Type: throw

Full Pipeline: Video -> MediaPipe -> BC -> Residual RL
"""

import gymnasium as gym
import numpy as np
from pathlib import Path
from datetime import datetime

from config import TASK_NAME, REWARD_CONFIG, BC_CONFIG, RESIDUAL_CONFIG, TRAINING_CONFIG
from baseline import BCBasePolicy, BCConfig
from residual import ResidualRLEnv, ResidualCurriculum
from reward import compute_reward


def main():
    from stable_baselines3 import TD3
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.monitor import Monitor

    print(f"Training: {TASK_NAME}")

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / TASK_NAME / f"{TASK_NAME}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = gym.make("AdroitHandRelocate-v1")
    env = Monitor(env, str(run_dir))

    # TD3 training
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=TRAINING_CONFIG["action_noise_sigma"] * np.ones(n_actions)
    )

    model = TD3(
        "MlpPolicy", env,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        buffer_size=TRAINING_CONFIG["buffer_size"],
        batch_size=TRAINING_CONFIG["batch_size"],
        action_noise=action_noise,
        verbose=1
    )

    model.learn(total_timesteps=TRAINING_CONFIG["total_timesteps"], progress_bar=True)
    model.save(run_dir / "td3_final")

    print(f"Saved to {run_dir}")
    env.close()


if __name__ == "__main__":
    main()
