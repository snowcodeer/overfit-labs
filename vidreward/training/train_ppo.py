import os
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from vidreward.training.env_wrapper import VidRewardWrapper
from vidreward.training.curriculum import PhaseLockCallback

def train():
    # 1. Setup Environment
    env_id = "AdroitHandRelocate-v1" 
    
    def make_env():
        env = gym.make(env_id)
        # Apply our reward wrapper
        env = VidRewardWrapper(env)
        return env

    env = DummyVecEnv([make_env])
    # Normalize observations and rewards for better RL convergence
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 2. Setup PPO Model
    # High-dimensional control usually benefits from a larger architecture
    policy_kwargs = dict(net_arch=[dict(pi=[256, 256, 256], vf=[256, 256, 256])])
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./tensorboard_logs/"
    )

    # 3. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/",
        name_prefix="vidreward_adroit"
    )
    curriculum_callback = PhaseLockCallback(verbose=1)

    # 4. Train
    print(f"Starting training on {env_id} with VidReward and Curriculum...")
    model.learn(
        total_timesteps=1000000,
        callback=[checkpoint_callback, curriculum_callback],
        progress_bar=True
    )

    # 5. Save final model
    model.save("vidreward_adroit_final")
    print("Training complete!")

if __name__ == "__main__":
    train()
