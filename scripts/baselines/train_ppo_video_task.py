"""
Pure RL Baseline for Video Task (M6)

Trains a standard PPO agent on the EXACT SAME task defined by the video.
- Same Reward Function (Phase-based / Dense)
- Same Initial Conditions (Grasp Pose / Object Pos)
- NO Residual Guidance (Pure RL controlling delta-qpos)

Usage:
    python scripts/baselines/train_ppo_video_task.py --video data/throw.mp4
"""

import argparse
import os
import sys
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.train_grasp_residual import extract_grasp_pose, GraspResidualEnv, PlotCallback

# The original make_env function is removed as it's replaced by a nested one in train()
# def make_env(args, grasp_qpos, target_pos, grasp_frame, task_type):
#     def _init():
#         base_env = gym.make("AdroitHandRelocate-v1")
#         # Reuse GraspResidualEnv but with NO transport trajectory
#         # This makes it a standard RL env where action -> delta_qpos
#         # base_qpos defaults to current_qpos in the wrapper when traj is None
#         env = GraspResidualEnv(
#             base_env,
#             grasp_qpos=grasp_qpos,
#             target_pos=target_pos,
#             transport_traj=None,  # <--- CRITICAL: Pure RL, no guidance
#             grasp_frame_idx=grasp_frame,
#             max_steps=args.max_steps,
#             residual_scale=0.1,   # Standard action scaling for Adroit
#             task_type=task_type
#         )
#         env = Monitor(env, info_keywords=('success', 'lifted'))
#         return env
#     return _init

def train(args):
    print(f"Training Pure RL Baseline for {args.video}")
    
    # 1. Load Task Config from Video
    grasp_qpos, target_pos, _, _, grasp_frame, _, task_type = extract_grasp_pose(args.video)
    print(f"Task Type: {task_type}")
    
    # 2. Create Envs
    # Create vectorized environment with ranks for Monitor logging
    def make_env(rank: int):
        def _init():
            env_id = "AdroitHandRelocate-v1"
            base_env = gym.make(env_id, max_episode_steps=args.max_steps)
            env = GraspResidualEnv(
                base_env,
                grasp_qpos=grasp_qpos,
                target_pos=target_pos,
                transport_traj=None,  # <--- CRITICAL: Pure RL, no guidance
                grasp_frame_idx=grasp_frame,
                max_steps=args.max_steps,
                residual_scale=0.1,   # Standard action scaling for Adroit
                task_type=task_type
            )
            # Log to run_dir/rank.monitor.csv
            monitor_path = os.path.join(args.output_dir, str(rank))
            env = Monitor(env, filename=monitor_path, info_keywords=('success', 'lifted'))
            return env
        return _init

    # Create envs manually to pass rank
    env_fns = [make_env(i) for i in range(args.num_envs)]
    if args.num_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)
        
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # 3. Train
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        tensorboard_log=os.path.join(args.output_dir, "tensorboard"),
        device="auto"
    )
    
    callbacks = [
        CheckpointCallback(save_freq=50000, save_path=os.path.join(args.output_dir, "checkpoints")),
        PlotCallback(os.path.join(args.output_dir, "plots"), os.path.basename(args.video), task_type=task_type)
    ]
    
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)
    model.save(os.path.join(args.output_dir, "final_model"))
    env.save(os.path.join(args.output_dir, "vec_normalize.pkl"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", default="runs/baselines/pure_rl")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100)
    
    args = parser.parse_args()
    train(args)
