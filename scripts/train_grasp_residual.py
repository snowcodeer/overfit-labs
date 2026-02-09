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
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from vidreward.extraction.mediapipe_tracker import MediaPipeTracker
from vidreward.retargeting.landmarks_to_angles import AdroitRetargeter
from vidreward.utils.video_io import VideoReader
from vidreward.extraction.vision import detect_rubiks_cube_classical
from vidreward.extraction.trajectory import ObjectTrajectory
from vidreward.phases.phase_detector import PhaseDetector


class PlotCallback(BaseCallback):
    """Track and plot training metrics."""

    def __init__(self, plot_dir: str, video_name: str, task_type: str = "pick", plot_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.plot_dir = plot_dir
        self.video_name = video_name
        self.task_type = task_type
        self.plot_freq = plot_freq
        os.makedirs(plot_dir, exist_ok=True)
        self.history_path = os.path.join(plot_dir, "history.pkl")

        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []
        self.distances = []
        self.contacts = []
        self.lifted = []
        self.timesteps_log = []

        self._load_history()

    def _load_history(self):
        import pickle
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, "rb") as f:
                    history = pickle.load(f)
                self.episode_rewards = history.get('episode_rewards', [])
                self.episode_lengths = history.get('episode_lengths', [])
                self.successes = history.get('successes', [])
                self.distances = history.get('distances', [])
                self.contacts = history.get('contacts', [])
                self.lifted = history.get('lifted', [])
                self.timesteps_log = history.get('timesteps_log', [])
                print(f"Loaded training history: {len(self.episode_rewards)} episodes.")
            except Exception as e:
                print(f"Warning: Could not load training history: {e}")

    def _save_history(self):
        import pickle
        history = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'successes': self.successes,
            'distances': self.distances,
            'contacts': self.contacts,
            'lifted': self.lifted,
            'timesteps_log': self.timesteps_log
        }
        with open(self.history_path, "wb") as f:
            pickle.dump(history, f)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.timesteps_log.append(self.num_timesteps)
                self.successes.append(1 if info.get("success", False) else 0)
                self.distances.append(info.get("dist_to_target", 1.0))
                self.contacts.append(info.get("n_contacts", 0))
                self.lifted.append(1 if info.get("lifted", False) else 0)

        if self.n_calls % 1000 == 0 and self.verbose > 0 and self.episode_rewards:
            recent_r = self.episode_rewards[-50:]
            recent_d = self.distances[-50:]
            recent_s = self.successes[-50:]
            recent_l = self.lifted[-50:]
            
            metric_name = "Lift" if self.task_type in ["pick", "slide"] else "Catch"
            
            print(f"Step {self.num_timesteps:,}: R={np.mean(recent_r):.1f}, "
                  f"Dist={np.mean(recent_d):.3f}, Succ={np.mean(recent_s):.1%}, {metric_name}={np.mean(recent_l):.1%}")

        if self.n_calls % self.plot_freq == 0 and len(self.episode_rewards) > 10:
            self._save_plots()
            self._save_history()

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
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards')
        ax.grid(True, alpha=0.3)

        if self.task_type in ["pick", "slide", "relocate"]:
            metric_label = 'Lift Rate'
            metric_title = 'Object Lift Rate'
        else:
            metric_label = 'Catch Rate'
            metric_title = 'Catch Success Rate'
            
        # Lift/Catch Rate (Crucial)
        ax = axes[0, 1]
        if len(self.lifted) > 50:
            w = 50
            rates = [np.mean(self.lifted[max(0,i-w):i]) for i in range(w, len(self.lifted)+1)]
            ax.plot(self.timesteps_log[w-1:], rates, color='orange', linewidth=2)
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_title)
        ax.grid(True, alpha=0.3)

        # Success rate
        ax = axes[1, 0]
        if len(self.successes) > 50:
            w = 50
            rates = [np.mean(self.successes[max(0,i-w):i]) for i in range(w, len(self.successes)+1)]
            ax.plot(self.timesteps_log[w-1:], rates, color='green', linewidth=2)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Success Rate')
        ax.set_title('Target Reached Rate')
        ax.grid(True, alpha=0.3)

        # Contacts
        ax = axes[1, 1]
        ax.plot(self.timesteps_log, self.contacts, alpha=0.3, color='purple')
        if len(self.contacts) > 50:
            w = 50
            smoothed = np.convolve(self.contacts, np.ones(w)/w, mode='valid')
            ax.plot(self.timesteps_log[w-1:], smoothed, color='purple', linewidth=2)
        ax.set_ylabel('Contacts')
        ax.set_title('Average Contacts')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'training.png'), dpi=100)
        plt.close()


def extract_grasp_pose(video_path: str):
    """
    Extract from video:
    - grasp_qpos: pose at grasp frame
    - target_pos: where object is released (in sim coords)
    - transport_traj: trajectory from grasp to release
    - hand_traj: Full MediaPipe trajectory
    """
    print(f"Loading video: {video_path}")
    reader = VideoReader(video_path)
    frames = list(reader.read_frames())

    print("Extracting hand trajectory...")
    tracker = MediaPipeTracker()
    hand_traj = tracker.process_frames(frames)
    tracker.close()

    # Simple object tracking
    h_vid, w_vid = frames[0].shape[:2]
    import cv2
    
    # Try detecting cube in first few frames
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

    import json
    
    # Check for Gemini analysis
    analysis_path = Path(video_path).parent.parent / "video_analysis.json" # Assumes data/video.mp4 -> root/video_analysis.json
    # Or check specific file matches
    # Better: check for "{video_stem}_analysis.json"
    analysis_path = Path(f"{Path(video_path).stem}_analysis.json")
    
    gemini_data = None
    task_type = "pick" # Default
    
    if analysis_path.exists():
        print(f"Loading Gemini analysis from {analysis_path}...")
        with open(analysis_path, 'r') as f:
            gemini_data = json.load(f)
            # Handle list or dict
            if isinstance(gemini_data, list):
                gemini_data = gemini_data[0]
    
    if gemini_data:
        print(f"Using Gemini metadata: {gemini_data}")
        grasp_frame = gemini_data.get('grasp_frame')
        release_frame = gemini_data.get('release_frame')
        task_type = gemini_data.get('task_type', "pick")
        
        # Handle "Catch" scenario (Toss then Catch -> Grasp > Release in timeline)
        if grasp_frame is not None and release_frame is not None:
            if grasp_frame > release_frame:
                print("WARNING: Grasp Frame > Release Frame. Assuming 'Catch' task.")
                print(f"Ignoring original release frame ({release_frame}) and using end of video.")
                release_frame = len(frames) - 1
                task_type = "catch"
    else:
        # Phase detection
        print("Detecting phases (Classical)...")
        obj_traj = ObjectTrajectory(centroids=obj_pos_vid_norm)
        phase_detector = PhaseDetector()
        phases = phase_detector.detect_phases(hand_traj, obj_traj)
    
        # Find grasp and release frames
        grasp_frame = 0
        release_frame = len(frames) - 1
        for phase in phases:
            if phase.label == "GRASP":
                grasp_frame = phase.start_frame
            if phase.label == "RELEASE":
                release_frame = phase.start_frame
        
    print(f"Grasp frame: {grasp_frame}")
    print(f"Release frame: {release_frame}")

    # Estimate release position
    # Ensure release_frame is within bounds
    release_frame = min(release_frame, len(frames)-1)
    grasp_frame = min(grasp_frame, len(frames)-1) # Safety
    
    release_pos_vid = obj_pos_vid_norm[release_frame]
    
    # Convert to sim coordinates (Approximate)
    SCALE = 0.6
    target_pos = np.array([
        (release_pos_vid[0] - 0.5) * SCALE,      
        -(release_pos_vid[1] - 0.5) * SCALE + 0.1,  
        0.15 
    ])
    
    # Retarget full trajectory
    print("Retargeting to joint angles...")
    retargeter = AdroitRetargeter()
    joint_traj = retargeter.retarget_sequence(hand_traj.landmarks)
    joint_traj = np.nan_to_num(joint_traj, nan=0.0)

    grasp_qpos = joint_traj[grasp_frame].copy()
    
    # Transport Traj: From Grasp to Release
    # If Catch: Grasp to End
    if release_frame <= grasp_frame:
         # Fallback if logic failed
         transport_traj = joint_traj[grasp_frame:].copy()
    else:
        transport_traj = joint_traj[grasp_frame:release_frame+1].copy()

    return grasp_qpos, target_pos, transport_traj, joint_traj, grasp_frame, hand_traj, task_type


def calibrate_grasp_pose(env, grasp_qpos: np.ndarray) -> np.ndarray:
    """
    Calibrate grasp pose using IK to ensure palm is near object.
    """
    import mujoco
    model = env.unwrapped.model
    data = env.unwrapped.data

    env.reset()
    obj_body_id = model.body("Object").id
    obj_pos = data.xpos[obj_body_id].copy()

    # Target: position palm above and behind cube
    GRASP_OFFSET = np.array([0.0, -0.08, 0.08])
    target_pos = obj_pos + GRASP_OFFSET

    palm_body_id = model.body("palm").id
    qpos_guess = grasp_qpos[:3].copy()

    # Iterative IK
    for _ in range(15):
        data.qpos[0:3] = qpos_guess
        data.qpos[3:30] = grasp_qpos[3:]
        mujoco.mj_forward(model, data)

        actual_palm = data.xpos[palm_body_id].copy()
        error = target_pos - actual_palm

        qpos_guess[0] -= error[0]
        qpos_guess[2] += error[1]
        qpos_guess[1] += error[2]

        if np.linalg.norm(error) < 0.002:
            break

    calibrated = grasp_qpos.copy()
    calibrated[0:3] = qpos_guess
    print(f"Calibrated grasp root: {qpos_guess}")
    
    return calibrated


class GraspResidualEnv(gym.Wrapper):
    """
    One-Shot Residual RL Environment.
    Starts at Calibrated Grasp Pose.
    Rewards Lifting and Transport.
    """
    def __init__(
        self,
        env,
        grasp_qpos: np.ndarray,
        target_pos: np.ndarray,
        transport_traj: np.ndarray = None,
        hand_traj=None,
        grasp_frame_idx: int = 0,
        max_steps: int = 150,
        residual_scale: float = 0.3,
    ):
        super().__init__(env)
        self.grasp_qpos = grasp_qpos
        self.target_pos = target_pos
        self.transport_traj = transport_traj
        self.hand_traj = hand_traj
        self.grasp_frame_idx = grasp_frame_idx
        self.max_steps = max_steps
        self.residual_scale = residual_scale

        self.steps = 0
        self.initial_obj_pos = None
        self.traj_idx = 0
        self.traj_offset = np.zeros(3)

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(30,), dtype=np.float32
        )
        
        # Obs: qpos(30) + obj_to_palm(3) + obj_to_target(3) + contact(1) + delta_origin(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32
        )

    def reset(self, **kwargs):
        import mujoco
        obs, info = self.env.reset(**kwargs)
        model = self.env.unwrapped.model
        data = self.env.unwrapped.data

        # 1. Get object pos
        obj_body_id = model.body("Object").id
        obj_pos = data.xpos[obj_body_id].copy()
        self.initial_obj_pos = obj_pos.copy()

        # 2. Calibrate start pose (Mini-IK at every reset for robustness)
        GRASP_OFFSET = np.array([0.0, -0.08, 0.08])
        target_hand_pos = obj_pos + GRASP_OFFSET
        
        qpos_guess = self.grasp_qpos[:3].copy()
        palm_body_id = model.body("palm").id

        for _ in range(10):
            data.qpos[0:3] = qpos_guess
            data.qpos[3:30] = self.grasp_qpos[3:]
            mujoco.mj_forward(model, data)
            actual_palm = data.xpos[palm_body_id].copy()
            error = target_hand_pos - actual_palm
            
            qpos_guess[0] -= error[0]
            qpos_guess[2] += error[1]
            qpos_guess[1] += error[2]
            
            if np.linalg.norm(error) < 0.002:
                break

        data.qpos[:3] = qpos_guess
        data.qpos[3:30] = self.grasp_qpos[3:]
        data.qvel[:] = 0

        # Settle
        for _ in range(5):
            mujoco.mj_step(model, data)

        # Set target visualization
        try:
            target_site_id = model.site("target").id
            model.site_pos[target_site_id] = self.target_pos.copy()
        except:
            pass

        self.steps = 0
        self.traj_idx = 0
        return self._get_obs(), info

    def step(self, action):
        self.steps += 1
        import mujoco

        current_qpos = self.env.unwrapped.data.qpos[:30].copy()

        # Base trajectory
        if self.transport_traj is not None and self.traj_idx < len(self.transport_traj):
            # Calculate offset on first frame
            if self.traj_idx == 0:
                self.traj_offset = current_qpos[:3] - self.transport_traj[0, :3]
            
            base_qpos = self.transport_traj[self.traj_idx].copy()
            base_qpos[:3] += self.traj_offset
            self.traj_idx += 1
        else:
            base_qpos = current_qpos

        # Residual
        residual = action * self.residual_scale
        target_qpos = base_qpos + residual
        target_qpos = np.clip(target_qpos, -2.5, 2.5)

        # Act
        self.env.unwrapped.data.ctrl[:30] = target_qpos
        for _ in range(5):
            mujoco.mj_step(self.env.unwrapped.model, self.env.unwrapped.data)

        reward, info = self._compute_reward()
        terminated = info.get('success', False)
        truncated = self.steps >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        data = self.env.unwrapped.data
        model = self.env.unwrapped.model

        qpos = data.qpos[:30].copy()
        obj_pos = data.xpos[model.body("Object").id].copy()
        palm_pos = data.xpos[model.body("palm").id].copy()
        
        obj_to_palm = obj_pos - palm_pos
        obj_to_target = self.target_pos - obj_pos
        n_contacts = min(data.ncon, 10)
        contact_indicator = np.array([n_contacts / 10.0])
        
        delta_origin = obj_pos - self.initial_obj_pos

        obs = np.concatenate([
            qpos, obj_to_palm, obj_to_target, contact_indicator, delta_origin
        ])
        return obs.astype(np.float32)

    def _compute_reward(self):
        data = self.env.unwrapped.data
        model = self.env.unwrapped.model
        
        obj_pos = data.xpos[model.body("Object").id].copy()
        dist_to_target = np.linalg.norm(obj_pos - self.target_pos)
        
        success = dist_to_target < 0.1
        is_lifted = obj_pos[2] > 0.08
        n_contacts = data.ncon
        
        reward = 0.0
        
        # 1. Contact
        reward += min(n_contacts, 5) * 0.1
        
        # 2. Lift
        if is_lifted:
            reward += 2.0
            # Transport bonus only if lifted
            reward += (1.0 - np.clip(dist_to_target, 0, 1)) * 5.0
        
        # 3. Success
        if success:
            reward += 50.0
            
        # Penalty
        if obj_pos[2] < 0.02: # Dropped
            reward -= 5.0
            
        info = {
            'success': success,
            'dist_to_target': dist_to_target,
            'n_contacts': n_contacts,
            'lifted': is_lifted
        }
        return reward, info


def train(args):
    from datetime import datetime
    import pickle
    import shutil

    video_name = Path(args.video).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_to_load = None
    if args.resume:
        source_dir = args.resume
        if not os.path.isdir(source_dir):
            raise FileNotFoundError(f"Resume directory {source_dir} not found.")
        
        # Create NEW run dir for branching
        run_name = f"{video_name}_{timestamp}_resumed"
        run_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        print(f"Resuming from {source_dir} -> Branched into {run_dir}")
        
        config_path = os.path.join(source_dir, "config.pkl")
        with open(config_path, "rb") as f:
            config = pickle.load(f)
        
        grasp_qpos = config['grasp_qpos']
        target_pos = config['target_pos']
        transport_traj = config['transport_traj']
        task_type = config.get('task_type', 'pick')

        # Copy history.pkl if it exists to preserve plotting
        src_history = os.path.join(source_dir, "plots", "history.pkl")
        dst_plots = os.path.join(run_dir, "plots")
        os.makedirs(dst_plots, exist_ok=True)
        dst_history = os.path.join(dst_plots, "history.pkl")
        if os.path.exists(src_history):
            shutil.copy(src_history, dst_history)
            print("Successfully copied training history.")

        # Save config to new directory
        with open(os.path.join(run_dir, "config.pkl"), "wb") as f:
            pickle.dump(config, f)
        
        # We still need hand_traj and grasp_frame for the wrapper
        _, _, _, _, grasp_frame, hand_traj, _ = extract_grasp_pose(args.video)

        # Find model to load
        model_path = os.path.join(source_dir, "td3_final.zip")
        if not os.path.exists(model_path):
            check_dir = os.path.join(source_dir, "checkpoints")
            if os.path.exists(check_dir):
                checkpoints = sorted([f for f in os.listdir(check_dir) if f.endswith(".zip")])
                if checkpoints:
                    model_path = os.path.join(check_dir, checkpoints[-1])
                else:
                    model_path = None
            else:
                model_path = None
        model_to_load = model_path
    else:
        run_name = f"{video_name}_{timestamp}"
        run_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        print(f"Training Residual RL for {video_name}")
        print(f"Directory: {run_dir}")

        # Extract & Calibrate
        grasp_qpos, target_pos, transport_traj, joint_traj, grasp_frame, hand_traj, task_type = extract_grasp_pose(args.video)
        
        # Init Env just to calibrate transport traj base
        base_env = gym.make("AdroitHandRelocate-v1")
        grasp_qpos = calibrate_grasp_pose(base_env, grasp_qpos)
        
        # Align transport traj start balanced with calibrated start
        offset = grasp_qpos[:3] - transport_traj[0, :3]
        transport_traj[:, :3] += offset
        base_env.close()

        # Save Config
        config = {
            'grasp_qpos': grasp_qpos,
            'target_pos': target_pos,
            'transport_traj': transport_traj,
            'residual_scale': args.residual_scale,
            'task_type': task_type
        }
        with open(os.path.join(run_dir, "config.pkl"), "wb") as f:
            pickle.dump(config, f)

    # Train Env
    def make_env():
        e = gym.make("AdroitHandRelocate-v1")
        e = GraspResidualEnv(
            e, grasp_qpos, target_pos, transport_traj, hand_traj, grasp_frame,
            max_steps=args.max_steps, residual_scale=args.residual_scale
        )
        e = Monitor(e, run_dir)
        return e

    env = make_env()

    if args.resume and model_to_load:
        print(f"Loading weights from {model_to_load}...")
        model = TD3.load(model_to_load, env=env)
    elif args.resume:
        print("No existing weights found in resume directory. Starting fresh in new branch.")
        model = None
    else:
        model = None

    if model is None:
        # TD3 initialization
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
            action_noise=action_noise,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1,
            tensorboard_log=os.path.join(run_dir, "tb"),
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=20000,
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix="td3"
    )

    plot_cb = PlotCallback(
        plot_dir=os.path.join(run_dir, "plots"),
        video_name=video_name,
        task_type=task_type,
        plot_freq=2000
    )

    print(f"Training for {args.timesteps} steps...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, plot_cb],
        progress_bar=True,
    )

    model.save(os.path.join(run_dir, "td3_final"))
    print("Training Complete!")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to demo video")
    parser.add_argument("--output-dir", default="runs/residual_pick3")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--residual-scale", type=float, default=0.3)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resume", type=str, help="Path to run directory to resume from")

    args = parser.parse_args()
    train(args)
