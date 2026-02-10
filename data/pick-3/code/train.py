"""
Residual RL Training Script for: pick-3
Task Type: slide

FULL PIPELINE:
1. Extract hand trajectory from video (MediaPipe)
2. Retarget to robot joint angles
3. Train BC base policy (single-demo with augmentation)
4. Train residual RL on top (TD3 learns corrections)

Modular files:
- config.py: All hyperparameters
- baseline.py: BC policy architecture
- residual.py: Residual RL wrapper
- reward.py: Custom reward function

To run:
  python train.py
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle

# Import from modular files
from config import (
    TASK_NAME, TASK_TYPE, MILESTONES,
    REWARD_CONFIG, BC_CONFIG, RESIDUAL_CONFIG, TRAINING_CONFIG
)
from baseline import BCTrainer, BCConfig, BCBasePolicy
from residual import ResidualRLEnv, ResidualCurriculum
from reward import compute_reward


def extract_trajectory_from_video(video_path: str):
    """
    Extract hand trajectory from video using MediaPipe.

    Returns:
        joint_trajectory: (T, 30) robot joint angles
        grasp_frame: frame where grasp begins
        target_pos: (3,) estimated target position
    """
    from vidreward.extraction.mediapipe_tracker import MediaPipeTracker
    from vidreward.retargeting.landmarks_to_angles import AdroitRetargeter
    from vidreward.utils.video_io import VideoReader

    print(f"Extracting trajectory from: {video_path}")

    # 1. Load video frames
    reader = VideoReader(video_path)
    frames = list(reader.read_frames())
    print(f"Loaded {len(frames)} frames")

    # 2. Extract hand landmarks with MediaPipe
    tracker = MediaPipeTracker()
    hand_traj = tracker.process_frames(frames)
    tracker.close()
    print(f"Extracted {len(hand_traj.landmarks)} hand poses")

    # 3. Retarget to robot joint angles
    retargeter = AdroitRetargeter()
    joint_trajectory = retargeter.retarget_sequence(hand_traj.landmarks)
    joint_trajectory = np.nan_to_num(joint_trajectory, nan=0.0)
    print(f"Retargeted to {joint_trajectory.shape} joint trajectory")

    # 4. Load analysis for grasp frame and target
    analysis_path = Path(video_path).parent / "analysis.json"
    grasp_frame = 0
    target_pos = np.array([0.0, 0.0, 0.2])

    if analysis_path.exists():
        import json
        with open(analysis_path) as f:
            analysis = json.load(f)
        grasp_frame = analysis.get("grasp_frame", 0)
        if "target_pos" in analysis:
            target_pos = np.array(analysis["target_pos"])

    return joint_trajectory, grasp_frame, target_pos


def main():
    from stable_baselines3 import TD3
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback

    print("=" * 60)
    print(f"RESIDUAL RL TRAINING: {TASK_NAME} ({TASK_TYPE})")
    print("=" * 60)

    # Paths
    task_dir = Path("data") / TASK_NAME
    video_path = task_dir / "video.mp4"

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / TASK_NAME / f"{TASK_NAME}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # =========================================================================
    # SAVE CODE FILES TO RUN DIRECTORY (for reproducibility + LLM context)
    # =========================================================================
    code_dir = run_dir / "code"
    code_dir.mkdir(exist_ok=True)

    # Copy all code files used for this training
    import shutil
    script_dir = Path(__file__).parent
    for code_file in ["config.py", "baseline.py", "residual.py", "reward.py", "train.py"]:
        src = script_dir / code_file
        if src.exists():
            shutil.copy(src, code_dir / code_file)
            print(f"Saved {code_file} to run directory")

    # Also save the full config as JSON for easy parsing
    import json
    config_snapshot = {
        "task_name": TASK_NAME,
        "task_type": TASK_TYPE,
        "milestones": MILESTONES,
        "reward_config": REWARD_CONFIG,
        "bc_config": BC_CONFIG,
        "residual_config": RESIDUAL_CONFIG,
        "training_config": TRAINING_CONFIG,
        "timestamp": timestamp,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2, default=str)
    print("Saved config.json snapshot")

    # =========================================================================
    # STEP 1: Extract trajectory from video
    # =========================================================================
    print("\n[Step 1/4] Extracting trajectory from video...")

    joint_traj, grasp_frame, target_pos = extract_trajectory_from_video(str(video_path))

    # Save trajectory
    with open(run_dir / "joint_trajectory.pkl", "wb") as f:
        pickle.dump({"trajectory": joint_traj, "grasp_frame": grasp_frame, "target_pos": target_pos}, f)

    # =========================================================================
    # STEP 2: Train BC base policy
    # =========================================================================
    print("\n[Step 2/4] Training BC base policy...")

    # Create env for data collection
    env = gym.make("AdroitHandRelocate-v1")

    # Collect state-action pairs by replaying trajectory in sim
    bc_config = BCConfig()
    trainer = BCTrainer(bc_config)
    states, actions = trainer.collect_data_from_sim(env, joint_traj)
    print(f"Collected {len(states)} state-action pairs")

    # Train BC policy
    bc_policy = trainer.train(states, actions, verbose=True)

    # Save BC policy
    import torch
    torch.save(bc_policy.state_dict(), run_dir / "bc_policy.pt")
    print(f"Saved BC policy to {run_dir / 'bc_policy.pt'}")

    env.close()

    # =========================================================================
    # STEP 3: Create Residual RL environment
    # =========================================================================
    print("\n[Step 3/4] Setting up Residual RL environment...")

    base_env = gym.make("AdroitHandRelocate-v1")

    # Wrap with residual RL
    env = ResidualRLEnv(
        env=base_env,
        base_policy=bc_policy,
        residual_scale=RESIDUAL_CONFIG["initial_scale"],
        residual_penalty_weight=RESIDUAL_CONFIG["penalty_weight"],
    )
    env = Monitor(env, str(run_dir))

    # Curriculum callback
    curriculum = ResidualCurriculum(
        env.env if hasattr(env, 'env') else env,  # Unwrap Monitor
        initial_scale=RESIDUAL_CONFIG["initial_scale"],
        final_scale=RESIDUAL_CONFIG["final_scale"],
        warmup_steps=RESIDUAL_CONFIG["warmup_steps"],
    )

    # Custom callback for curriculum + logging
    class CurriculumCallback(BaseCallback):
        def __init__(self, curriculum):
            super().__init__()
            self.curriculum = curriculum

        def _on_step(self):
            metrics = self.curriculum.on_step()
            for k, v in metrics.items():
                self.logger.record(f"curriculum/{k}", v)
            return True

    # =========================================================================
    # STEP 4: Train TD3 residual policy
    # =========================================================================
    print("\n[Step 4/4] Training TD3 residual policy...")

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=TRAINING_CONFIG["action_noise_sigma"] * np.ones(n_actions)
    )

    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        buffer_size=TRAINING_CONFIG["buffer_size"],
        learning_starts=TRAINING_CONFIG["learning_starts"],
        batch_size=TRAINING_CONFIG["batch_size"],
        tau=TRAINING_CONFIG.get("tau", 0.005),
        gamma=TRAINING_CONFIG.get("gamma", 0.99),
        action_noise=action_noise,
        policy_kwargs=dict(net_arch=TRAINING_CONFIG["policy_arch"]),
        tensorboard_log=str(run_dir / "tb"),
        verbose=1
    )

    print(f"\nStarting TD3 training for {TRAINING_CONFIG['total_timesteps']} steps...")
    print(f"Residual scale: {RESIDUAL_CONFIG['initial_scale']} -> {RESIDUAL_CONFIG['final_scale']}")

    model.learn(
        total_timesteps=TRAINING_CONFIG["total_timesteps"],
        callback=CurriculumCallback(curriculum),
        progress_bar=True
    )

    # Save final model
    model.save(run_dir / "td3_residual_final")
    print(f"\nSaved TD3 model to {run_dir / 'td3_residual_final.zip'}")

    # =========================================================================
    # Done!
    # =========================================================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Results saved to: {run_dir}")
    print(f"  - bc_policy.pt: BC base policy")
    print(f"  - td3_residual_final.zip: Trained residual policy")
    print(f"  - tb/: TensorBoard logs")

    env.close()


if __name__ == "__main__":
    main()
