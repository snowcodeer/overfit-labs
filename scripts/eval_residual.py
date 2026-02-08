"""
Evaluate trained residual RL model with video output.

Usage:
    python scripts/eval_residual.py --run runs/residual_rl/pick-3_20250209_143052 --episodes 5
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import argparse
import os
import cv2
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import TD3
from vidreward.utils.video_io import VideoWriter


def load_run_config(run_dir: str):
    """Load the grasp_qpos, target_pos, transport_traj from run."""
    import pickle

    config_path = os.path.join(run_dir, "config.pkl")
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            return pickle.load(f)
    return None


class GraspResidualEnv(gym.Wrapper):
    """Same env as training - copy for eval."""

    def __init__(
        self,
        env,
        grasp_qpos: np.ndarray,
        target_pos: np.ndarray,
        transport_traj: np.ndarray = None,
        max_steps: int = 150,
        residual_scale: float = 0.4,
    ):
        super().__init__(env)

        self.grasp_qpos = grasp_qpos
        self.target_pos = target_pos
        self.transport_traj = transport_traj
        self.max_steps = max_steps
        self.residual_scale = residual_scale

        self.steps = 0
        self.initial_obj_pos = None
        self.traj_idx = 0

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(30,), dtype=np.float32
        )
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

        obj_body_id = self.env.unwrapped.model.body("Object").id
        self.initial_obj_pos = self.env.unwrapped.data.xpos[obj_body_id].copy()

        self.steps = 0
        self.traj_idx = 0

        return self._get_obs(), info

    def step(self, action):
        self.steps += 1
        import mujoco

        current_qpos = self.env.unwrapped.data.qpos[:30].copy()

        if self.transport_traj is not None and self.traj_idx < len(self.transport_traj):
            base_qpos = self.transport_traj[self.traj_idx]
            self.traj_idx += 1
        else:
            base_qpos = current_qpos

        residual = action * self.residual_scale
        target_qpos = base_qpos + residual
        target_qpos = np.clip(target_qpos, -2.5, 2.5)

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

        obj_body_id = model.body("Object").id
        palm_body_id = model.body("palm").id
        obj_pos = data.xpos[obj_body_id].copy()
        palm_pos = data.xpos[palm_body_id].copy()
        obj_to_palm = obj_pos - palm_pos
        obj_to_target = self.target_pos - obj_pos

        n_contacts = min(data.ncon, 10)
        contact_indicator = np.array([n_contacts / 10.0])

        obs = np.concatenate([qpos, obj_to_palm, obj_to_target, contact_indicator])
        return obs.astype(np.float32)

    def _compute_reward(self):
        data = self.env.unwrapped.data
        model = self.env.unwrapped.model

        obj_body_id = model.body("Object").id
        palm_body_id = model.body("palm").id
        obj_pos = data.xpos[obj_body_id].copy()
        palm_pos = data.xpos[palm_body_id].copy()

        dist_to_target = np.linalg.norm(obj_pos - self.target_pos)
        obj_palm_dist = np.linalg.norm(obj_pos - palm_pos)
        obj_height = obj_pos[2]
        lifted = obj_height > 0.06

        info = {}
        reward = 0.0

        n_contacts = data.ncon
        contact_reward = min(n_contacts, 8) * 0.2
        reward += contact_reward

        grasp_bonus = np.exp(-10.0 * obj_palm_dist) * 2.0
        reward += grasp_bonus

        if lifted and n_contacts >= 3:
            reward += 3.0

        initial_dist = np.linalg.norm(self.initial_obj_pos - self.target_pos)
        if lifted:
            progress = (initial_dist - dist_to_target) / (initial_dist + 1e-6)
            reward += progress * 5.0
            reward += np.exp(-3.0 * dist_to_target) * 2.0

        if dist_to_target < 0.05:
            reward += 20.0
            info['success'] = True
        elif dist_to_target < 0.10:
            reward += 5.0

        if obj_pos[2] < 0.02:
            reward -= 5.0
        if obj_palm_dist > 0.3:
            reward -= 2.0

        info['dist_to_target'] = dist_to_target
        info['obj_palm_dist'] = obj_palm_dist
        info['n_contacts'] = n_contacts
        info['lifted'] = lifted
        info['obj_pos'] = obj_pos.copy()

        return reward, info


def evaluate(args):
    run_dir = args.run

    print("=" * 60)
    print("Evaluating Residual RL Model")
    print("=" * 60)
    print(f"Run: {run_dir}")

    # Load model
    if args.checkpoint:
        # Load specific checkpoint
        model_path = os.path.join(run_dir, "checkpoints", f"td3_{args.checkpoint}_steps.zip")
        if not os.path.exists(model_path):
            model_path = os.path.join(run_dir, "checkpoints", f"td3_{args.checkpoint}_steps")
    else:
        # Load final model - try multiple naming conventions
        candidates = [
            os.path.join(run_dir, "td3_final.zip"),
            os.path.join(run_dir, "td3_final"),
            os.path.join(run_dir, "grasp_residual_td3_final.zip"),
            os.path.join(run_dir, "grasp_residual_td3_final"),
        ]
        model_path = None
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break

    if model_path is None or not os.path.exists(model_path):
        print(f"ERROR: Model not found in {run_dir}")
        print("Available files:")
        for f in os.listdir(run_dir):
            print(f"  {f}")
        return

    print(f"Loading model: {model_path}")
    model = TD3.load(model_path)

    # Load config
    config_path = os.path.join(run_dir, "config.pkl")
    if os.path.exists(config_path):
        import pickle
        with open(config_path, "rb") as f:
            config = pickle.load(f)
        grasp_qpos = config['grasp_qpos']
        target_pos = config['target_pos']
        transport_traj = config.get('transport_traj', None)
        video_name = config.get('video_name', 'unknown')
    elif args.video:
        # Re-extract from video
        print(f"No config.pkl found. Re-extracting from video: {args.video}")
        from vidreward.extraction.mediapipe_tracker import MediaPipeTracker
        from vidreward.retargeting.landmarks_to_angles import AdroitRetargeter
        from vidreward.utils.video_io import VideoReader
        from vidreward.extraction.vision import detect_rubiks_cube_classical
        from vidreward.extraction.trajectory import ObjectTrajectory
        from vidreward.phases.phase_detector import PhaseDetector

        reader = VideoReader(args.video)
        frames = list(reader.read_frames())
        tracker = MediaPipeTracker()
        hand_traj = tracker.process_frames(frames)
        tracker.close()

        h_vid, w_vid = frames[0].shape[:2]
        import cv2
        bbox = None
        for i in range(min(50, len(frames))):
            bbox = detect_rubiks_cube_classical(frames[i])
            if bbox: break
        curr_box = bbox if bbox else (w_vid//2-40, h_vid//2-40, 80, 80)

        obj_pos_vid_norm = np.zeros((len(frames), 2))
        sim_tracker = cv2.TrackerCSRT_create()
        sim_tracker.init(frames[0], curr_box)
        for i in range(len(frames)):
            success, box = sim_tracker.update(frames[i])
            if success: curr_box = box
            bx, by, bw, bh = curr_box
            obj_pos_vid_norm[i] = [(bx+bw/2)/w_vid, (by+bh/2)/h_vid]

        obj_traj = ObjectTrajectory(centroids=obj_pos_vid_norm)
        phase_detector = PhaseDetector()
        phases = phase_detector.detect_phases(hand_traj, obj_traj)

        grasp_frame, release_frame = 0, len(frames)-1
        for phase in phases:
            if phase.label == "GRASP": grasp_frame = phase.start_frame
            if phase.label == "RELEASE": release_frame = phase.start_frame

        release_pos_vid = obj_pos_vid_norm[release_frame]
        SCALE = 0.6
        target_pos = np.array([
            (release_pos_vid[0] - 0.5) * SCALE,
            -(release_pos_vid[1] - 0.5) * SCALE + 0.1,
            0.15
        ])

        retargeter = AdroitRetargeter()
        joint_traj = retargeter.retarget_sequence(hand_traj.landmarks)
        joint_traj = np.nan_to_num(joint_traj, nan=0.0)
        grasp_qpos = joint_traj[grasp_frame].copy()
        transport_traj = joint_traj[grasp_frame:release_frame+1].copy()

        # Calibrate
        temp_env = gym.make("AdroitHandRelocate-v1")
        temp_env.reset()
        import mujoco
        mj_model = temp_env.unwrapped.model
        mj_data = temp_env.unwrapped.data
        obj_body_id = mj_model.body("Object").id
        obj_pos = mj_data.xpos[obj_body_id].copy()
        palm_body_id = mj_model.body("palm").id

        GRASP_OFFSET = np.array([0.0, -0.06, 0.06])
        target_palm = obj_pos + GRASP_OFFSET
        qpos_guess = np.array([0.0, 0.0, 0.0])
        for _ in range(10):
            mj_data.qpos[0:3] = qpos_guess
            mj_data.qpos[3:30] = grasp_qpos[3:]
            mujoco.mj_forward(mj_model, mj_data)
            error = target_palm - mj_data.xpos[palm_body_id]
            qpos_guess[0] -= error[0]
            qpos_guess[2] += error[1]
            qpos_guess[1] += error[2]
            if np.linalg.norm(error) < 0.001: break
        grasp_qpos[:3] = qpos_guess
        offset = grasp_qpos[:3] - transport_traj[0, :3]
        transport_traj[:, :3] += offset
        temp_env.close()

        video_name = Path(args.video).stem
        # Store video release position for visualization
        release_pos_vid_px = release_pos_vid  # normalized [0,1]
    else:
        print("ERROR: config.pkl not found.")
        print("Provide --video to re-extract config from video.")
        return

    # Get release position in video coords for visualization
    if 'release_pos_vid_px' not in dir():
        release_pos_vid_px = None

    print(f"Video: {video_name}")
    print(f"Target position: {target_pos}")

    # Create env
    base_env = gym.make("AdroitHandRelocate-v1", render_mode="rgb_array")
    env = GraspResidualEnv(
        base_env,
        grasp_qpos=grasp_qpos,
        target_pos=target_pos,
        transport_traj=transport_traj,
        max_steps=args.max_steps,
        residual_scale=args.residual_scale,
    )

    # Eval loop
    print(f"\nRunning {args.episodes} evaluation episodes...")

    # Save debug screenshot of initial state
    obs, _ = env.reset()
    debug_frame = env.env.render()
    debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR)

    # Add debug info
    h, w = debug_frame.shape[:2]
    cv2.putText(debug_frame, "DEBUG: Initial State", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(debug_frame, f"Target (green sphere): ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})",
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(debug_frame, f"Video: {video_name}", (10, 85),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(debug_frame, "Green sphere = where cube should go (from video)", (10, h-20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    debug_path = os.path.join(run_dir, "debug_target.png")
    cv2.imwrite(debug_path, debug_frame)
    print(f"Debug screenshot saved: {debug_path}")

    all_frames = []
    successes = 0

    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        ep_frames = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            done = term or trunc

            # Render frame
            frame = env.env.render()
            if frame is not None:
                # Add overlay
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                h, w = frame.shape[:2]

                # Info overlay
                dist = info.get('dist_to_target', 0)
                contacts = info.get('n_contacts', 0)
                lifted = info.get('lifted', False)
                success = info.get('success', False)
                obj_pos = info.get('obj_pos', np.zeros(3))

                color = (0, 255, 0) if success else ((0, 165, 255) if lifted else (255, 255, 255))

                cv2.putText(frame, f"Ep {ep+1}/{args.episodes}", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Dist: {dist:.3f}m", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"Contacts: {contacts}", (10, 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Lifted: {lifted}", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if success:
                    cv2.putText(frame, "SUCCESS!", (10, 130),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Debug info panel (bottom left) - env's green sphere shows target
                cv2.rectangle(frame, (5, h-60), (200, h-5), (0, 0, 0), -1)
                cv2.putText(frame, f"Dist to target: {dist:.3f}m",
                           (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                cv2.putText(frame, f"Contacts: {contacts} | Lifted: {lifted}",
                           (10, h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                ep_frames.append(frame)

        success = info.get('success', False)
        successes += int(success)
        lifted = info.get('lifted', False)
        dist = info.get('dist_to_target', 0)

        print(f"  Ep {ep+1}: reward={total_reward:.1f}, dist={dist:.3f}, "
              f"lifted={lifted}, success={success}")

        all_frames.extend(ep_frames)

        # Add separator frames between episodes
        if ep < args.episodes - 1:
            sep_frame = np.zeros_like(ep_frames[0])
            cv2.putText(sep_frame, f"Episode {ep+2}", (200, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            all_frames.extend([sep_frame] * 15)  # 0.5 sec at 30fps

    print(f"\nSuccess rate: {successes}/{args.episodes} = {successes/args.episodes:.1%}")

    # Save video
    if all_frames:
        output_path = os.path.join(run_dir, f"eval_{datetime.now().strftime('%H%M%S')}.mp4")
        print(f"\nSaving video to: {output_path}")

        h, w = all_frames[0].shape[:2]
        with VideoWriter(output_path, fps=30.0, width=w, height=h) as writer:
            for frame in all_frames:
                writer.write_frame(frame)

        print(f"Video saved! ({len(all_frames)} frames)")

    env.close()

    return successes / args.episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Path to run directory")
    parser.add_argument("--video", type=str, default=None,
                        help="Video path to re-extract config (if config.pkl missing)")
    parser.add_argument("--checkpoint", type=int, default=None,
                        help="Checkpoint steps to load (e.g., 100000). Default: final model")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--residual-scale", type=float, default=0.4)

    args = parser.parse_args()
    evaluate(args)
