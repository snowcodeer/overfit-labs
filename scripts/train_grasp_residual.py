"""
Grasp-Focused Residual RL

The video trajectory handles APPROACH perfectly.
This script trains a residual policy ONLY for the GRASP phase.

Key insight:
- Base policy: kinematic trajectory from video (approach + initial grasp pose)
- Residual RL: learns finger corrections to actually grip the object
- Much simpler than full task learning!
"""

import gymnasium as gym
import gymnasium_robotics  # Required to register Adroit envs
import numpy as np
import torch
import argparse
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3.common.callbacks import BaseCallback


class PlotCallback(BaseCallback):
    """Track and plot training metrics."""

    def __init__(self, plot_dir: str, video_name: str, plot_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.plot_dir = plot_dir
        self.video_name = video_name
        self.plot_freq = plot_freq
        os.makedirs(plot_dir, exist_ok=True)

        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []
        self.distances = []
        self.contacts = []
        self.timesteps_log = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.timesteps_log.append(self.num_timesteps)
                self.successes.append(1 if info.get("success", False) else 0)
                self.distances.append(info.get("dist_to_target", 1.0))
                self.contacts.append(info.get("n_contacts", 0))

        if self.n_calls % 1000 == 0 and self.verbose > 0 and self.episode_rewards:
            recent_r = self.episode_rewards[-50:] if len(self.episode_rewards) >= 50 else self.episode_rewards
            recent_d = self.distances[-50:] if len(self.distances) >= 50 else self.distances
            recent_s = self.successes[-50:] if len(self.successes) >= 50 else self.successes
            print(f"Step {self.num_timesteps:,}: reward={np.mean(recent_r):.1f}, "
                  f"dist={np.mean(recent_d):.3f}, success={np.mean(recent_s):.1%}")

        if self.n_calls % self.plot_freq == 0 and len(self.episode_rewards) > 10:
            self._save_plots()

        return True

    def _save_plots(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Residual RL: {self.video_name} - {self.num_timesteps:,} steps', fontsize=14)

        # Rewards
        ax = axes[0, 0]
        ax.plot(self.timesteps_log, self.episode_rewards, alpha=0.3, color='blue')
        if len(self.episode_rewards) > 50:
            w = 50
            smoothed = np.convolve(self.episode_rewards, np.ones(w)/w, mode='valid')
            ax.plot(self.timesteps_log[w-1:], smoothed, color='blue', linewidth=2)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Episode Rewards')
        ax.grid(True, alpha=0.3)

        # Distance to target
        ax = axes[0, 1]
        ax.plot(self.timesteps_log, self.distances, alpha=0.3, color='orange')
        if len(self.distances) > 50:
            w = 50
            smoothed = np.convolve(self.distances, np.ones(w)/w, mode='valid')
            ax.plot(self.timesteps_log[w-1:], smoothed, color='orange', linewidth=2)
        ax.axhline(y=0.05, color='green', linestyle='--', label='Success threshold')
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Distance to Target')
        ax.set_title('Final Distance to Target')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Success rate
        ax = axes[1, 0]
        if len(self.successes) > 50:
            w = 50
            rates = [np.mean(self.successes[max(0,i-w):i]) for i in range(w, len(self.successes)+1)]
            ax.plot(self.timesteps_log[w-1:], rates, color='green', linewidth=2)
            ax.fill_between(self.timesteps_log[w-1:], 0, rates, alpha=0.3, color='green')
        ax.set_ylim(0, 1)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate (rolling 50 eps)')
        ax.grid(True, alpha=0.3)

        # Contacts
        ax = axes[1, 1]
        ax.plot(self.timesteps_log, self.contacts, alpha=0.3, color='purple')
        if len(self.contacts) > 50:
            w = 50
            smoothed = np.convolve(self.contacts, np.ones(w)/w, mode='valid')
            ax.plot(self.timesteps_log[w-1:], smoothed, color='purple', linewidth=2)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Contacts')
        ax.set_title('Finger-Object Contacts')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'training.png'), dpi=150)
        plt.close()

from vidreward.extraction.mediapipe_tracker import MediaPipeTracker
from vidreward.retargeting.landmarks_to_angles import AdroitRetargeter
from vidreward.utils.video_io import VideoReader
from vidreward.extraction.vision import detect_rubiks_cube_classical
from vidreward.extraction.trajectory import ObjectTrajectory
from vidreward.phases.phase_detector import PhaseDetector


class GraspResidualEnv(gym.Wrapper):
    """
    Environment that:
    1. Resets to the GRASP frame pose (from video trajectory)
    2. Agent outputs residual adjustments
    3. Reward: grasp + transport to TARGET (release position from video)

    This is MUCH easier than learning from scratch:
    - Hand is already positioned correctly
    - Target comes from video demonstration
    """

    def __init__(
        self,
        env,
        grasp_qpos: np.ndarray,
        target_pos: np.ndarray,  # Release position from video
        transport_traj: np.ndarray = None,  # Optional: trajectory from grasp to release
        max_steps: int = 150,
        residual_scale: float = 0.3,
    ):
        super().__init__(env)

        self.grasp_qpos = grasp_qpos  # Starting pose at grasp frame
        self.target_pos = target_pos  # Where to release (from video)
        self.transport_traj = transport_traj  # Base trajectory for transport
        self.max_steps = max_steps
        self.residual_scale = residual_scale

        self.steps = 0
        self.initial_obj_pos = None
        self.traj_idx = 0  # Current index in transport trajectory

        # Control full arm + fingers for transport
        # Action: residual corrections to base trajectory
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(30,), dtype=np.float32
        )

        # Observation space: qpos(30) + obj_to_palm(3) + obj_to_target(3) + contact(1) = 37
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32
        )

    def reset(self, **kwargs):
        import mujoco

        obs, info = self.env.reset(**kwargs)

        model = self.env.unwrapped.model
        data = self.env.unwrapped.data

        # Fix target position to match video's release position
        # The target is stored in model.site_pos, NOT env.goal
        target_site_id = model.site("target").id
        model.site_pos[target_site_id] = self.target_pos.copy()

        # Set to grasp pose
        data.qpos[:30] = self.grasp_qpos
        data.qvel[:] = 0

        # Step physics to settle and update site positions
        for _ in range(10):
            mujoco.mj_step(model, data)

        # Record initial object position
        obj_body_id = self.env.unwrapped.model.body("Object").id
        self.initial_obj_pos = self.env.unwrapped.data.xpos[obj_body_id].copy()

        self.steps = 0
        self.traj_idx = 0

        return self._get_obs(), info

    def step(self, action):
        self.steps += 1
        import mujoco

        current_qpos = self.env.unwrapped.data.qpos[:30].copy()

        # Get base action from trajectory (if available)
        if self.transport_traj is not None and self.traj_idx < len(self.transport_traj):
            base_qpos = self.transport_traj[self.traj_idx]
            self.traj_idx += 1
        else:
            base_qpos = current_qpos

        # Apply residual correction
        residual = action * self.residual_scale
        target_qpos = base_qpos + residual

        # Clip to joint limits
        target_qpos = np.clip(target_qpos, -2.5, 2.5)

        # Use actuator control (respects physics)
        self.env.unwrapped.data.ctrl[:30] = target_qpos

        # Step physics
        for _ in range(5):
            mujoco.mj_step(self.env.unwrapped.model, self.env.unwrapped.data)

        # Compute reward
        reward, info = self._compute_reward()

        # Check termination
        terminated = info.get('success', False)
        truncated = self.steps >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        """Get observation focused on task-relevant features."""
        data = self.env.unwrapped.data
        model = self.env.unwrapped.model

        # Full joint positions
        qpos = data.qpos[:30].copy()

        # Object position relative to palm
        obj_body_id = model.body("Object").id
        palm_body_id = model.body("palm").id
        obj_pos = data.xpos[obj_body_id].copy()
        palm_pos = data.xpos[palm_body_id].copy()
        obj_to_palm = obj_pos - palm_pos

        # Object position relative to target
        obj_to_target = self.target_pos - obj_pos

        # Contact info (simplified)
        n_contacts = min(data.ncon, 10)
        contact_indicator = np.array([n_contacts / 10.0])

        obs = np.concatenate([
            qpos,             # 30
            obj_to_palm,      # 3
            obj_to_target,    # 3
            contact_indicator # 1
        ])

        return obs.astype(np.float32)

    def _compute_reward(self):
        """Reward for grasping and reaching target. Emphasizes stable grasp."""
        data = self.env.unwrapped.data
        model = self.env.unwrapped.model

        obj_body_id = model.body("Object").id
        palm_body_id = model.body("palm").id
        obj_pos = data.xpos[obj_body_id].copy()
        palm_pos = data.xpos[palm_body_id].copy()

        # Distance to target
        dist_to_target = np.linalg.norm(obj_pos - self.target_pos)
        initial_dist = np.linalg.norm(self.initial_obj_pos - self.target_pos)

        # Object-palm distance (for grasp)
        obj_palm_dist = np.linalg.norm(obj_pos - palm_pos)

        # Object height (lifted?)
        obj_height = obj_pos[2]
        lifted = obj_height > 0.06  # Above table

        info = {}
        reward = 0.0

        # 1. GRASP REWARD (most important!) - Contact + close to palm
        n_contacts = data.ncon
        contact_reward = min(n_contacts, 8) * 0.2  # Up to 1.6 for good contact
        reward += contact_reward

        # Bonus for object being close to palm (in hand)
        grasp_bonus = np.exp(-10.0 * obj_palm_dist) * 2.0  # Up to 2.0
        reward += grasp_bonus

        # 2. LIFT REWARD - Object above table while grasped
        if lifted and n_contacts >= 3:
            reward += 3.0  # Big bonus for lifting with grasp

        # 3. TRANSPORT REWARD - Progress toward target (only if lifted)
        if lifted:
            progress = (initial_dist - dist_to_target) / (initial_dist + 1e-6)
            reward += progress * 5.0

            # Distance-based shaping
            reward += np.exp(-3.0 * dist_to_target) * 2.0

        # 4. SUCCESS - Object at target
        if dist_to_target < 0.05:
            reward += 20.0
            info['success'] = True
        elif dist_to_target < 0.10:
            reward += 5.0  # Close bonus

        # 5. PENALTIES
        # Dropped object
        if obj_pos[2] < 0.02:
            reward -= 5.0
        # Object flew away from palm (lost grasp)
        if obj_palm_dist > 0.3:
            reward -= 2.0

        info['dist_to_target'] = dist_to_target
        info['obj_palm_dist'] = obj_palm_dist
        info['n_contacts'] = n_contacts
        info['lifted'] = lifted
        info['obj_pos'] = obj_pos.copy()

        return reward, info


def extract_grasp_pose(video_path: str):
    """
    Extract from video:
    - grasp_qpos: pose at grasp frame
    - target_pos: where object is released (in sim coords)
    - transport_traj: trajectory from grasp to release
    """

    print(f"Loading video: {video_path}")
    reader = VideoReader(video_path)
    frames = list(reader.read_frames())

    print("Extracting hand trajectory...")
    tracker = MediaPipeTracker()
    hand_traj = tracker.process_frames(frames)
    tracker.close()

    # Object tracking for phase detection
    h_vid, w_vid = frames[0].shape[:2]
    import cv2

    bbox = None
    for i in range(min(50, len(frames))):
        bbox = detect_rubiks_cube_classical(frames[i])
        if bbox:
            break

    if bbox is None:
        curr_box = (w_vid//2 - 40, h_vid//2 - 40, 80, 80)
    else:
        curr_box = bbox

    obj_pos_vid_norm = np.zeros((len(frames), 2))
    sim_tracker = cv2.TrackerCSRT_create()
    sim_tracker.init(frames[0], curr_box)

    for i in range(len(frames)):
        success, box = sim_tracker.update(frames[i])
        if success:
            curr_box = box
        bx, by, bw, bh = curr_box
        cx, cy = bx + bw/2, by + bh/2
        obj_pos_vid_norm[i] = [cx / w_vid, cy / h_vid]

    # Phase detection
    print("Detecting phases...")
    obj_traj = ObjectTrajectory(centroids=obj_pos_vid_norm)
    phase_detector = PhaseDetector()
    phases = phase_detector.detect_phases(hand_traj, obj_traj)

    # Find grasp and release frames
    grasp_frame = 0
    release_frame = len(frames) - 1
    for phase in phases:
        print(f"  {phase.label}: frames {phase.start_frame}-{phase.end_frame}")
        if phase.label == "GRASP":
            grasp_frame = phase.start_frame
        if phase.label == "RELEASE":
            release_frame = phase.start_frame

    print(f"Grasp frame: {grasp_frame}")
    print(f"Release frame: {release_frame}")

    # Get release position (where object ends up)
    release_pos_vid = obj_pos_vid_norm[release_frame]
    print(f"Release position (video normalized): {release_pos_vid}")

    # Convert to sim coordinates
    # Video: (0,0) top-left, (1,1) bottom-right
    # Sim: X left-right, Y forward-back, Z up-down
    # Workspace approx: X[-0.3, 0.3], Y[-0.1, 0.3], Z[0.035, 0.3]
    SCALE = 0.6
    target_pos = np.array([
        (release_pos_vid[0] - 0.5) * SCALE,      # X: center at 0
        -(release_pos_vid[1] - 0.5) * SCALE + 0.1,  # Y: flip, offset
        0.15  # Z: reasonable height above table
    ])
    print(f"Target position (sim): {target_pos}")

    # Retarget full trajectory
    print("Retargeting to joint angles...")
    retargeter = AdroitRetargeter()
    joint_traj = retargeter.retarget_sequence(hand_traj.landmarks)
    joint_traj = np.nan_to_num(joint_traj, nan=0.0)

    grasp_qpos = joint_traj[grasp_frame].copy()

    # Transport trajectory: from grasp to release
    transport_traj = joint_traj[grasp_frame:release_frame+1].copy()
    print(f"Transport trajectory: {len(transport_traj)} frames")

    return grasp_qpos, target_pos, transport_traj, joint_traj, grasp_frame


def calibrate_grasp_pose(env, grasp_qpos: np.ndarray) -> np.ndarray:
    """
    Calibrate the grasp pose to position hand correctly over object.
    Similar to replay_m1.py calibration.
    """
    import mujoco

    model = env.unwrapped.model
    data = env.unwrapped.data

    env.reset()

    # Get object position
    obj_body_id = model.body("Object").id
    obj_pos = data.xpos[obj_body_id].copy()

    # Target: position palm above and behind cube
    GRASP_OFFSET = np.array([0.0, -0.06, 0.06])
    target_pos = obj_pos + GRASP_OFFSET

    palm_body_id = model.body("palm").id
    qpos_guess = np.array([0.0, 0.0, 0.0])

    # Iterative calibration
    for _ in range(10):
        data.qpos[0:3] = qpos_guess
        data.qpos[3:30] = grasp_qpos[3:]
        mujoco.mj_forward(model, data)

        actual_palm = data.xpos[palm_body_id].copy()
        error = target_pos - actual_palm

        qpos_guess[0] -= error[0]
        qpos_guess[2] += error[1]
        qpos_guess[1] += error[2]

        if np.linalg.norm(error) < 0.001:
            break

    calibrated = grasp_qpos.copy()
    calibrated[0:3] = qpos_guess

    print(f"Calibrated grasp position: {qpos_guess}")

    return calibrated


def train(args):
    from datetime import datetime

    # Create run folder: video_name + datetime
    video_name = Path(args.video).stem  # e.g., "pick-3"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{video_name}_{timestamp}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 60)
    print("Residual RL: Video-Guided Relocate")
    print("=" * 60)
    print(f"Video: {args.video}")
    print(f"Run dir: {run_dir}")
    print("=" * 60)

    # Extract grasp pose and target from video
    grasp_qpos, target_pos, transport_traj, joint_traj, grasp_frame = extract_grasp_pose(args.video)

    # Create environment
    print("\nCreating environment...")
    base_env = gym.make("AdroitHandRelocate-v1")

    # Calibrate pose
    grasp_qpos = calibrate_grasp_pose(base_env, grasp_qpos)

    # Also calibrate transport trajectory positions
    print("Calibrating transport trajectory...")
    offset = grasp_qpos[:3] - transport_traj[0, :3]
    transport_traj[:, :3] += offset  # Apply same offset to whole trajectory

    base_env.close()

    # Save config for eval script
    import pickle
    config = {
        'grasp_qpos': grasp_qpos,
        'target_pos': target_pos,
        'transport_traj': transport_traj,
        'video_name': video_name,
        'video_path': args.video,
        'residual_scale': args.residual_scale,
    }
    with open(os.path.join(run_dir, "config.pkl"), "wb") as f:
        pickle.dump(config, f)
    print(f"Config saved to {run_dir}/config.pkl")

    # Create residual env
    base_env = gym.make("AdroitHandRelocate-v1")
    env = GraspResidualEnv(
        base_env,
        grasp_qpos=grasp_qpos,
        target_pos=target_pos,
        transport_traj=transport_traj,
        residual_scale=args.residual_scale,
        max_steps=args.max_steps,
    )

    # Quick test
    print("\nTesting environment...")
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action shape: {env.action_space.shape}")
    print(f"Target position: {target_pos}")

    for i in range(5):
        action = env.action_space.sample() * 0.1  # Small random action
        obs, reward, term, trunc, info = env.step(action)
        print(f"  Step {i}: reward={reward:.3f}, dist={info['dist_to_target']:.4f}, contacts={info['n_contacts']}")

    env.close()

    # Train with TD3 (same as original residual RL paper)
    print("\n" + "=" * 60)
    print("Training TD3 (as in original Residual RL paper)...")
    print("=" * 60)

    try:
        from stable_baselines3 import TD3
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.noise import NormalActionNoise
    except ImportError as e:
        print(f"ERROR: stable-baselines3 not installed: {e}")
        print("Run: pip install stable-baselines3")
        return

    # Fresh env for training
    base_env = gym.make("AdroitHandRelocate-v1")
    env = GraspResidualEnv(
        base_env,
        grasp_qpos=grasp_qpos,
        target_pos=target_pos,
        transport_traj=transport_traj,
        residual_scale=args.residual_scale,
        max_steps=args.max_steps,
    )
    env = Monitor(env, run_dir)

    # Action noise for exploration (important for TD3)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        buffer_size=100_000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        action_noise=action_noise,
        policy_kwargs=dict(net_arch=[256, 256]),  # Larger network for TD3
        verbose=1,
        tensorboard_log=os.path.join(run_dir, "tb"),
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,  # Every 100k steps
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix="td3"
    )

    plot_cb = PlotCallback(
        plot_dir=os.path.join(run_dir, "plots"),
        video_name=video_name,
        plot_freq=5000,
        verbose=1
    )

    print(f"\nTraining for {args.timesteps} steps...")
    print(f"TD3 is better for fine-grained continuous control!")
    print(f"Plots: {run_dir}/plots/training.png")

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, plot_cb],
        progress_bar=True,
    )

    model.save(os.path.join(run_dir, "td3_final"))
    print(f"\nDone! Model saved to {run_dir}/td3_final")

    # Quick eval
    print("\nEvaluating...")
    successes = 0
    for ep in range(10):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            done = term or trunc

        success = info.get('success', False)
        successes += int(success)
        lifted = info.get('lifted', False)
        contacts = info.get('n_contacts', 0)
        print(f"  Ep {ep+1}: reward={total_reward:.1f}, dist={info['dist_to_target']:.3f}, "
              f"contacts={contacts}, lifted={lifted}, success={success}")

    print(f"\nSuccess rate: {successes}/10 = {successes*10}%")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to demo video")
    parser.add_argument("--output-dir", default="runs/residual_rl")
    parser.add_argument("--timesteps", type=int, default=200000,
                        help="Training steps")
    parser.add_argument("--residual-scale", type=float, default=0.4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)
