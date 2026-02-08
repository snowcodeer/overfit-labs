"""
One-Shot Residual RL from Video

Train a residual RL policy from a single video demonstration.

Based on: "Residual Reinforcement Learning for Robot Control" (Johannink et al., 2018)

Pipeline:
1. Extract hand trajectory from video (MediaPipe)
2. Retarget to robot joint angles (AdroitRetargeter)
3. Collect state-action pairs via sim replay
4. Train BC base policy with augmentation
5. Train residual RL policy (PPO) on top

Usage:
    python scripts/train_residual_rl.py --video data/pick-rubiks-cube.mp4
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import gymnasium as gym
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vidreward.training.bc_policy import (
    BCConfig,
    BCTrainer,
    BCBasePolicy,
    create_bc_base_policy_from_video
)
from vidreward.training.residual_env import (
    ResidualRLEnv,
    ResidualCurriculumCallback
)
from vidreward.extraction.mediapipe_tracker import MediaPipeTracker
from vidreward.retargeting.landmarks_to_angles import AdroitRetargeter
from vidreward.utils.video_io import VideoReader


def train_residual_rl(
    video_path: str,
    output_dir: str = "runs/residual_rl",
    total_timesteps: int = 500_000,
    seed: int = 42,
    # BC params
    bc_epochs: int = 1000,
    bc_hidden_dims: tuple = (128, 128),
    bc_augment_factor: int = 50,
    # Residual params
    initial_residual_scale: float = 0.1,
    final_residual_scale: float = 0.4,
    residual_warmup_steps: int = 150_000,
    residual_penalty_weight: float = 0.1,
    # PPO params
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    ent_coef: float = 0.01,
    # Misc
    eval_freq: int = 10_000,
    save_freq: int = 50_000,
    verbose: int = 1,
):
    """
    Main training function for one-shot residual RL from video.

    Args:
        video_path: Path to demonstration video
        output_dir: Directory for logs and checkpoints
        total_timesteps: Total RL training steps
        seed: Random seed
        bc_*: Behavioral cloning parameters
        initial/final_residual_scale: Curriculum bounds for residual authority
        residual_warmup_steps: Steps to reach final residual scale
        residual_penalty_weight: Penalty for large residuals
        learning_rate, n_steps, etc: PPO hyperparameters
        eval_freq: Evaluation frequency
        save_freq: Checkpoint save frequency
        verbose: Verbosity level
    """
    # Setup
    np.random.seed(seed)
    torch.manual_seed(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("One-Shot Residual RL from Video")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Output: {run_dir}")
    print(f"Total timesteps: {total_timesteps}")
    print("=" * 60)

    # =========================================================================
    # Phase 1: Video Processing & BC Training
    # =========================================================================
    print("\n[Phase 1] Extracting demonstration from video...")

    # Load video
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    reader = VideoReader(video_path)
    frames = list(reader.read_frames())
    print(f"  Loaded {len(frames)} frames")

    # Extract hand trajectory
    print("  Extracting hand landmarks...")
    tracker = MediaPipeTracker()
    hand_traj = tracker.process_frames(frames)
    tracker.close()
    print(f"  Extracted {len(hand_traj.landmarks)} hand poses")

    # Retarget to joint angles
    print("  Retargeting to robot joints...")
    retargeter = AdroitRetargeter()
    joint_trajectory = retargeter.retarget_sequence(hand_traj.landmarks)
    joint_trajectory = np.nan_to_num(joint_trajectory, nan=0.0)
    print(f"  Joint trajectory shape: {joint_trajectory.shape}")

    # Save trajectory for reference
    np.save(run_dir / "joint_trajectory.npy", joint_trajectory)

    # =========================================================================
    # Phase 2: Collect BC Training Data via Sim Replay
    # =========================================================================
    print("\n[Phase 2] Collecting training data via sim replay...")

    # Create base environment
    base_env = gym.make("AdroitHandRelocate-v1", render_mode="rgb_array")
    base_env.reset(seed=seed)

    # Collect state-action pairs
    bc_config = BCConfig(
        hidden_dims=bc_hidden_dims,
        epochs=bc_epochs,
        augment_samples_per_frame=bc_augment_factor,
    )
    trainer = BCTrainer(bc_config)
    states, actions = trainer.collect_data_from_sim(base_env, joint_trajectory)
    print(f"  Collected {len(states)} state-action pairs")
    print(f"  State dim: {states.shape[1]}, Action dim: {actions.shape[1]}")

    # =========================================================================
    # Phase 3: Train BC Base Policy
    # =========================================================================
    print("\n[Phase 3] Training BC base policy...")

    bc_policy = trainer.train(states, actions, verbose=(verbose > 0))

    # Save BC policy
    bc_path = run_dir / "bc_policy.pt"
    torch.save({
        'state_dict': bc_policy.state_dict(),
        'state_dim': bc_policy.state_dim,
        'action_dim': bc_policy.action_dim,
        'state_mean': bc_policy.state_mean.numpy(),
        'state_std': bc_policy.state_std.numpy(),
        'action_scale': bc_policy.action_scale.numpy(),
    }, bc_path)
    print(f"  Saved BC policy to {bc_path}")

    # Evaluate BC policy
    print("  Evaluating BC policy...")
    bc_rewards = evaluate_bc_policy(base_env, bc_policy, n_episodes=5)
    print(f"  BC policy reward: {np.mean(bc_rewards):.2f} +/- {np.std(bc_rewards):.2f}")

    base_env.close()

    # =========================================================================
    # Phase 4: Create Residual RL Environment
    # =========================================================================
    print("\n[Phase 4] Setting up residual RL environment...")

    # Create fresh env for RL training
    base_env = gym.make("AdroitHandRelocate-v1")

    residual_env = ResidualRLEnv(
        env=base_env,
        base_policy=bc_policy,
        residual_scale=initial_residual_scale,
        residual_penalty_weight=residual_penalty_weight,
    )

    # Create curriculum callback
    curriculum = ResidualCurriculumCallback(
        env=residual_env,
        initial_scale=initial_residual_scale,
        final_scale=final_residual_scale,
        warmup_steps=residual_warmup_steps,
    )

    print(f"  Initial residual scale: {initial_residual_scale}")
    print(f"  Final residual scale: {final_residual_scale}")
    print(f"  Warmup steps: {residual_warmup_steps}")

    # =========================================================================
    # Phase 5: Train Residual Policy with PPO
    # =========================================================================
    print("\n[Phase 5] Training residual policy with PPO...")

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import (
            EvalCallback,
            CheckpointCallback,
            BaseCallback
        )
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        print("ERROR: stable-baselines3 not installed.")
        print("Install with: pip install stable-baselines3")
        return

    # Wrap for logging
    residual_env = Monitor(residual_env, str(run_dir / "monitor"))

    # Custom callback for curriculum
    class CurriculumCallback(BaseCallback):
        def __init__(self, curriculum, verbose=0):
            super().__init__(verbose)
            self.curriculum = curriculum

        def _on_step(self):
            metrics = self.curriculum.on_step()
            for key, value in metrics.items():
                self.logger.record(key, value)
            return True

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        residual_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        ent_coef=ent_coef,
        policy_kwargs=dict(
            net_arch=[256, 256, 256],
            ortho_init=True,
        ),
        verbose=verbose,
        tensorboard_log=str(run_dir / "tensorboard"),
        seed=seed,
    )

    # Callbacks
    callbacks = [
        CurriculumCallback(curriculum, verbose=verbose),
        CheckpointCallback(
            save_freq=save_freq,
            save_path=str(run_dir / "checkpoints"),
            name_prefix="residual_ppo"
        ),
    ]

    # Train
    print(f"  Starting training for {total_timesteps} steps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_path = run_dir / "residual_ppo_final"
    model.save(str(final_path))
    print(f"  Saved final model to {final_path}")

    # =========================================================================
    # Phase 6: Evaluate Final Policy
    # =========================================================================
    print("\n[Phase 6] Evaluating final policy...")

    final_rewards = evaluate_residual_policy(residual_env, model, n_episodes=10)
    print(f"  Final policy reward: {np.mean(final_rewards):.2f} +/- {np.std(final_rewards):.2f}")

    # Compare to BC-only
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"BC-only reward:       {np.mean(bc_rewards):.2f} +/- {np.std(bc_rewards):.2f}")
    print(f"Residual RL reward:   {np.mean(final_rewards):.2f} +/- {np.std(final_rewards):.2f}")
    improvement = np.mean(final_rewards) - np.mean(bc_rewards)
    print(f"Improvement:          {improvement:+.2f}")
    print("=" * 60)

    residual_env.close()

    return model, bc_policy


def evaluate_bc_policy(
    env: gym.Env,
    policy: BCBasePolicy,
    n_episodes: int = 5
) -> list:
    """Evaluate BC policy (no residual)."""
    from vidreward.training.bc_policy import extract_features_from_sim_state
    import mujoco

    rewards = []
    model = env.unwrapped.model
    data = env.unwrapped.data

    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Get state features
            state_features = extract_features_from_sim_state(model, data)

            # Get BC action (delta)
            delta_action = policy.get_action(state_features)

            # Apply as delta to current qpos
            current_qpos = data.qpos[:30].copy()
            target_qpos = current_qpos + delta_action
            target_qpos = np.clip(
                target_qpos,
                env.action_space.low,
                env.action_space.high
            )

            obs, reward, terminated, truncated, info = env.step(target_qpos)
            episode_reward += reward
            done = terminated or truncated

        rewards.append(episode_reward)

    return rewards


def evaluate_residual_policy(env, model, n_episodes: int = 10) -> list:
    """Evaluate trained residual policy."""
    rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        rewards.append(episode_reward)

    return rewards


def load_and_evaluate(run_dir: str, video_path: str = None):
    """Load a saved model and evaluate it."""
    run_dir = Path(run_dir)

    # Load BC policy
    bc_checkpoint = torch.load(run_dir / "bc_policy.pt")
    bc_policy = BCBasePolicy(
        state_dim=bc_checkpoint['state_dim'],
        action_dim=bc_checkpoint['action_dim'],
    )
    bc_policy.load_state_dict(bc_checkpoint['state_dict'])
    bc_policy.state_mean = torch.FloatTensor(bc_checkpoint['state_mean'])
    bc_policy.state_std = torch.FloatTensor(bc_checkpoint['state_std'])
    bc_policy.action_scale = torch.FloatTensor(bc_checkpoint['action_scale'])

    # Load PPO model
    from stable_baselines3 import PPO

    ppo_path = run_dir / "residual_ppo_final"
    model = PPO.load(str(ppo_path))

    # Create env
    base_env = gym.make("AdroitHandRelocate-v1", render_mode="human")
    env = ResidualRLEnv(
        env=base_env,
        base_policy=bc_policy,
        residual_scale=0.4,  # Final scale
        residual_penalty_weight=0.01,
    )

    # Evaluate
    rewards = evaluate_residual_policy(env, model, n_episodes=5)
    print(f"Evaluation reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-Shot Residual RL from Video")

    # Required
    parser.add_argument("--video", type=str, required=True,
                        help="Path to demonstration video")

    # Output
    parser.add_argument("--output-dir", type=str, default="runs/residual_rl",
                        help="Output directory")

    # Training
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # BC params
    parser.add_argument("--bc-epochs", type=int, default=1000,
                        help="BC training epochs")
    parser.add_argument("--bc-augment", type=int, default=50,
                        help="Augmentation factor per frame")

    # Residual params
    parser.add_argument("--initial-scale", type=float, default=0.1,
                        help="Initial residual scale")
    parser.add_argument("--final-scale", type=float, default=0.4,
                        help="Final residual scale")
    parser.add_argument("--warmup-steps", type=int, default=150_000,
                        help="Curriculum warmup steps")
    parser.add_argument("--residual-penalty", type=float, default=0.1,
                        help="Residual penalty weight")

    # Misc
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity level")

    # Evaluation mode
    parser.add_argument("--eval", type=str, default=None,
                        help="Evaluate a saved run (path to run dir)")

    args = parser.parse_args()

    if args.eval:
        load_and_evaluate(args.eval, args.video)
    else:
        train_residual_rl(
            video_path=args.video,
            output_dir=args.output_dir,
            total_timesteps=args.timesteps,
            seed=args.seed,
            bc_epochs=args.bc_epochs,
            bc_augment_factor=args.bc_augment,
            initial_residual_scale=args.initial_scale,
            final_residual_scale=args.final_scale,
            residual_warmup_steps=args.warmup_steps,
            residual_penalty_weight=args.residual_penalty,
            verbose=args.verbose,
        )
