"""
Video-guided environment for learning from demonstration.

Uses the original Adroit reward structure (proven to work) plus
trajectory guidance from video demonstration.
"""
import gymnasium as gym
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class VideoDemo:
    """Demonstration extracted from video."""
    joint_angles: np.ndarray      # (num_frames, 30) retargeted joints
    grasp_frame: int              # Frame where grasp occurs
    release_frame: int            # Frame where release occurs
    target_pos: np.ndarray        # (3,) where object should end up
    object_start_pos: np.ndarray  # (3,) where object starts
    num_frames: int

    @classmethod
    def from_video(cls, video_path: str, mirror: bool = False) -> "VideoDemo":
        """Load demonstration from video file."""
        import cv2
        from vidreward.extraction.mediapipe_tracker import MediaPipeTracker
        from vidreward.retargeting.landmarks_to_angles import AdroitRetargeter
        from vidreward.utils.video_io import VideoReader
        from vidreward.extraction.release_detector import detect_release_position

        # Load video
        reader = VideoReader(video_path)
        frames = list(reader.read_frames())

        if mirror:
            frames = [cv2.flip(f, 1) for f in frames]

        # Extract hand trajectory
        tracker = MediaPipeTracker()
        hand_traj = tracker.process_frames(frames)
        tracker.close()

        # Retarget to joint angles
        retargeter = AdroitRetargeter()
        joint_angles = retargeter.retarget_sequence(hand_traj.landmarks)

        # Get release info
        release_info = detect_release_position(video_path, mirror=mirror)

        return cls(
            joint_angles=joint_angles,
            grasp_frame=release_info.grasp_frame,
            release_frame=release_info.release_frame,
            target_pos=release_info.release_pos_sim,
            object_start_pos=release_info.start_pos_sim,
            num_frames=len(frames),
        )


class VideoGuidedRelocateEnv(gym.Wrapper):
    """
    Wraps AdroitHandRelocate with video demonstration guidance.

    Key features:
    1. Sets target position from video (where object is released)
    2. Provides reference trajectory for guidance
    3. Uses original Adroit dense reward (proven to work)
    4. Adds small trajectory bonus to guide exploration
    """

    def __init__(
        self,
        env: gym.Env,
        demo: VideoDemo,
        traj_bonus_weight: float = 0.1,
        use_demo_target: bool = True,
    ):
        super().__init__(env)
        self.demo = demo
        self.traj_bonus_weight = traj_bonus_weight
        self.use_demo_target = use_demo_target

        # Current step in episode (maps to demo frame)
        self.step_count = 0
        self.frame_ratio = 1.0  # Sim steps per video frame

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0

        # Override target position with demo target
        if self.use_demo_target:
            self._set_target_from_demo()

        info["demo_target"] = self.demo.target_pos.copy()
        info["demo_frame"] = 0

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Step the environment (gets original Adroit reward)
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add trajectory tracking bonus
        demo_frame = min(
            int(self.step_count * self.frame_ratio),
            self.demo.num_frames - 1
        )
        ref_qpos = self.demo.joint_angles[demo_frame]

        # Current joint positions (first 30 elements of obs)
        current_qpos = obs[:30] if len(obs) >= 30 else np.zeros(30)

        # Trajectory bonus: reward for being close to demo
        # Only for finger joints (6-29), not base position
        joint_error = np.linalg.norm(current_qpos[6:30] - ref_qpos[6:30])
        traj_bonus = np.exp(-0.5 * joint_error)  # Max 1.0

        reward += self.traj_bonus_weight * traj_bonus

        # Update info
        info["demo_frame"] = demo_frame
        info["traj_bonus"] = traj_bonus
        info["joint_error"] = joint_error

        self.step_count += 1

        return obs, reward, terminated, truncated, info

    def _set_target_from_demo(self):
        """Set the environment's target to match the demo release position."""
        try:
            # Access the underlying environment's target
            unwrapped = self.env.unwrapped
            target_site_id = unwrapped._model_names.site_name2id["target"]

            # Set target position from demo
            unwrapped.model.site_pos[target_site_id] = self.demo.target_pos.copy()
        except Exception as e:
            print(f"Warning: Could not set target from demo: {e}")

    def get_demo_action(self, step: int) -> np.ndarray:
        """Get the demonstration action for a given step (for residual learning)."""
        demo_frame = min(
            int(step * self.frame_ratio),
            self.demo.num_frames - 1
        )
        return self.demo.joint_angles[demo_frame].copy()

    def set_traj_bonus_weight(self, weight: float):
        """Set trajectory bonus weight (for curriculum learning)."""
        self.traj_bonus_weight = weight


def make_video_guided_env(
    video_path: str,
    mirror: bool = False,
    traj_bonus_weight: float = 0.1,
    render_mode: str = None,
) -> VideoGuidedRelocateEnv:
    """
    Create a video-guided relocate environment.

    Args:
        video_path: Path to demonstration video
        mirror: Mirror video for left->right hand conversion
        traj_bonus_weight: Weight for trajectory following bonus
        render_mode: Gymnasium render mode

    Returns:
        Wrapped environment
    """
    import gymnasium_robotics

    # Load demo
    print(f"Loading demo from: {video_path}")
    demo = VideoDemo.from_video(video_path, mirror=mirror)
    print(f"  Frames: {demo.num_frames}")
    print(f"  Grasp frame: {demo.grasp_frame}")
    print(f"  Release frame: {demo.release_frame}")
    print(f"  Target pos: {demo.target_pos}")

    # Create base environment
    env = gym.make("AdroitHandRelocate-v1", render_mode=render_mode)

    # Wrap with video guidance
    env = VideoGuidedRelocateEnv(
        env,
        demo=demo,
        traj_bonus_weight=traj_bonus_weight,
    )

    return env
