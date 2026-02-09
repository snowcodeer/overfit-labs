"""
Train an RL agent to perform manipulation learned from video demonstration.

Uses:
1. Original Adroit dense reward (proven to work)
2. Target position from video (where object is released)
3. Small trajectory bonus to guide exploration
4. Phase-locked curriculum learning
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from vidreward.training.video_guided_env import (
    VideoGuidedRelocateEnv,
    VideoDemo,
    make_video_guided_env,
)


class TrainingMetricsCallback(BaseCallback):
    """Track and plot training metrics."""

    def __init__(self, plot_dir: str, plot_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.plot_dir = plot_dir
        self.plot_freq = plot_freq
        os.makedirs(plot_dir, exist_ok=True)

        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []
        self.timesteps_log = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.timesteps_log.append(self.num_timesteps)

                # Track success
                success = info.get("success", False)
                self.successes.append(1 if success else 0)

        # Console logging
        if self.n_calls % 2000 == 0 and self.verbose > 0:
            if self.episode_rewards:
                recent = self.episode_rewards[-100:]
                print(f"Step {self.num_timesteps:,}: "
                      f"reward={np.mean(recent):.1f}, "
                      f"success={np.mean(self.successes[-100:]):.1%}")

        # Save plots
        if self.n_calls % self.plot_freq == 0 and len(self.episode_rewards) > 10:
            self._save_plots()

        return True

    def _save_plots(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Training Progress - {self.num_timesteps:,} steps', fontsize=14)

        # Rewards
        ax = axes[0, 0]
        ax.plot(self.timesteps_log, self.episode_rewards, alpha=0.3, color='blue')
        if len(self.episode_rewards) > 50:
            window = 50
            smoothed = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            ax.plot(self.timesteps_log[window-1:], smoothed, color='blue', linewidth=2)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Episode Rewards')
        ax.grid(True, alpha=0.3)

        # Success rate
        ax = axes[0, 1]
        if len(self.successes) > 50:
            window = 100
            success_rate = []
            for i in range(window, len(self.successes)):
                success_rate.append(np.mean(self.successes[i-window:i]))
            ax.plot(self.timesteps_log[window:], success_rate, color='green', linewidth=2)
            ax.fill_between(self.timesteps_log[window:], 0, success_rate, alpha=0.3, color='green')
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate (rolling 100 episodes)')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # Episode length
        ax = axes[1, 0]
        ax.plot(self.timesteps_log, self.episode_lengths, alpha=0.3, color='orange')
        if len(self.episode_lengths) > 50:
            window = 50
            smoothed = np.convolve(self.episode_lengths, np.ones(window)/window, mode='valid')
            ax.plot(self.timesteps_log[window-1:], smoothed, color='orange', linewidth=2)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Episode Length')
        ax.set_title('Episode Length')
        ax.grid(True, alpha=0.3)

        # Reward histogram
        ax = axes[1, 1]
        recent = self.episode_rewards[-200:] if len(self.episode_rewards) > 200 else self.episode_rewards
        ax.hist(recent, bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(recent), color='red', linestyle='--', label=f'Mean: {np.mean(recent):.1f}')
        ax.set_xlabel('Reward')
        ax.set_ylabel('Count')
        ax.set_title('Recent Rewards Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'training.png'), dpi=150)
        plt.close()


class CurriculumCallback(BaseCallback):
    """
    Curriculum that adjusts trajectory bonus weight over time.

    Starts with higher trajectory following, gradually reduces to let
    RL discover better solutions.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.stages = [
            # (timesteps, traj_bonus_weight)
            (0,       0.3),   # Early: follow demo more
            (100_000, 0.2),   # Mid: balance
            (300_000, 0.1),   # Late: rely more on task reward
            (500_000, 0.05),  # Final: minimal guidance
        ]
        self.current_stage = 0

    def _on_step(self) -> bool:
        # Find current stage
        stage_idx = 0
        for i, (threshold, _) in enumerate(self.stages):
            if self.num_timesteps >= threshold:
                stage_idx = i

        if stage_idx != self.current_stage:
            self.current_stage = stage_idx
            weight = self.stages[stage_idx][1]
            print(f"\n>>> Curriculum stage {stage_idx + 1}: traj_bonus_weight = {weight}\n")

            # Update envs
            try:
                self.training_env.env_method("set_traj_bonus_weight", weight)
            except:
                pass

        return True


def make_env(demo: VideoDemo, traj_bonus_weight: float, rank: int = 0):
    """Factory for creating environments."""
    def _init():
        import gymnasium as gym
        import gymnasium_robotics

        env = gym.make("AdroitHandRelocate-v1")
        env = VideoGuidedRelocateEnv(env, demo, traj_bonus_weight=traj_bonus_weight)
        env = Monitor(env)
        return env
    return _init


def train(args):
    print("=" * 60)
    print("Video-Guided RL Training")
    print("=" * 60)

    # Load demonstration
    print(f"\nLoading demonstration from: {args.video}")
    demo = VideoDemo.from_video(args.video, mirror=args.mirror)
    print(f"  Frames: {demo.num_frames}")
    print(f"  Grasp frame: {demo.grasp_frame}")
    print(f"  Target position: {demo.target_pos}")

    # Create environments
    print(f"\nCreating {args.num_envs} environments...")
    if args.num_envs > 1:
        env = SubprocVecEnv([
            make_env(demo, args.traj_weight, rank=i)
            for i in range(args.num_envs)
        ])
    else:
        env = DummyVecEnv([make_env(demo, args.traj_weight)])

    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Create model
    print("\nCreating PPO model...")
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=os.path.join(args.output_dir, "tensorboard"),
        device=args.device,
    )

    # Callbacks
    plot_dir = os.path.join(args.output_dir, "plots")
    callbacks = [
        TrainingMetricsCallback(plot_dir=plot_dir, plot_freq=10000, verbose=1),
        CheckpointCallback(
            save_freq=25000,
            save_path=os.path.join(args.output_dir, "checkpoints"),
            name_prefix="model",
        ),
    ]

    if args.curriculum:
        callbacks.append(CurriculumCallback(verbose=1))

    # Train
    print(f"\nStarting training for {args.timesteps:,} timesteps...")
    print(f"  Device: {args.device}")
    print(f"  Trajectory weight: {args.traj_weight}")
    print(f"  Curriculum: {args.curriculum}")
    print(f"  Output: {args.output_dir}")
    print()

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save
    model.save(os.path.join(args.output_dir, "model_final"))
    env.save(os.path.join(args.output_dir, "vec_normalize.pkl"))

    print(f"\nTraining complete!")
    print(f"  Model: {args.output_dir}/model_final.zip")
    print(f"  Plots: {plot_dir}/training.png")


def evaluate(args):
    """Evaluate trained model."""
    import time

    print(f"Loading model from: {args.model}")
    model = PPO.load(args.model)

    print(f"Loading demo from: {args.video}")
    demo = VideoDemo.from_video(args.video, mirror=args.mirror)

    env = make_video_guided_env(
        args.video,
        mirror=args.mirror,
        traj_bonus_weight=0.0,
        render_mode="human",
    )

    successes = 0
    for ep in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            time.sleep(0.02)

        success = info.get("success", False)
        successes += int(success)
        print(f"Episode {ep+1}: reward={total_reward:.1f}, success={success}")

    print(f"\nSuccess rate: {successes}/{args.episodes} ({successes/args.episodes:.1%})")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train from video demonstration")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument("--video", required=True, help="Path to demo video")
    train_parser.add_argument("--mirror", action="store_true", help="Mirror video (left hand)")
    train_parser.add_argument("--timesteps", type=int, default=1_000_000)
    train_parser.add_argument("--num-envs", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--traj-weight", type=float, default=0.1)
    train_parser.add_argument("--curriculum", action="store_true", default=True)
    train_parser.add_argument("--no-curriculum", dest="curriculum", action="store_false")
    train_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    train_parser.add_argument("--output-dir", default="./runs/video_guided/")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model")
    eval_parser.add_argument("--model", required=True, help="Path to model")
    eval_parser.add_argument("--video", required=True, help="Path to demo video")
    eval_parser.add_argument("--mirror", action="store_true")
    eval_parser.add_argument("--episodes", type=int, default=10)

    args = parser.parse_args()

    if args.command == "train":
        os.makedirs(args.output_dir, exist_ok=True)
        train(args)
    else:
        evaluate(args)
