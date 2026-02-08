import gymnasium as gym
import numpy as np
from typing import Dict, Any, Optional
from vidreward.rewards.composer import RewardComposer
from vidreward.rewards.primitives import ReachReward, GraspReward, LiftReward, SmoothnessPenalty

class VidRewardWrapper(gym.Wrapper):
    """
    Gymnasium wrapper for Adroit environments that replaces the default reward
    with VidReward's phase-dependent rewards.
    """
    def __init__(self, env: gym.Env, composer: Optional[RewardComposer] = None):
        super().__init__(env)
        if composer is None:
            # Initialize a default composer if none provided
            self.composer = RewardComposer()
            self.composer.add_primitive("reach", ReachReward())
            self.composer.add_primitive("grasp", GraspReward())
            self.composer.add_primitive("lift", LiftReward())
            self.composer.add_primitive("smoothness", SmoothnessPenalty())
        else:
            self.composer = composer
            
        self.current_phase = "IDLE"
        self.locked_phase: Optional[str] = None # If set, only reward this phase

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 1. Update Phase Heuristics (Mapping Sim State to Phases)
        self.current_phase = self._detect_sim_phase(obs, info)
        
        # 2. Extract state features for reward primitives
        # State mapping from Adroit obs/info
        state_features = self._extract_state_features(obs, info)
        
        # 3. Compute VidReward
        phase_to_reward = self.locked_phase if self.locked_phase else self.current_phase
        vid_reward = self.composer.compute_reward(state_features, info, phase_to_reward)
        
        # 4. Update info for logging
        info["vid_reward"] = vid_reward
        info["current_phase"] = self.current_phase
        
        return obs, vid_reward, terminated, truncated, info

    def _detect_sim_phase(self, obs, info) -> str:
        """
        Simple heuristic logic to detect the current phase in simulation.
        This mirrors the logic in PhaseDetector but for sim state.
        """
        # Distances normally provided in info for Adroit Hand envs
        dist = info.get("hand_to_obj_dist", 1.0)
        is_grasped = info.get("is_grasped", False)
        obj_pos = info.get("obj_pos", np.zeros(3))
        target_pos = info.get("target_pos", np.zeros(3))
        
        # Transition Logic
        if self.current_phase == "IDLE":
            if dist < 0.3: return "APPROACH"
        elif self.current_phase == "APPROACH":
            if is_grasped: return "GRASP"
            if dist > 0.4: return "IDLE"
        elif self.current_phase == "GRASP":
            # If object is lifted/moving towards target
            lift_dist = np.linalg.norm(obj_pos - target_pos)
            if lift_dist < 0.1: return "TRANSPORT" # Getting close to target
            if not is_grasped: return "APPROACH"
        elif self.current_phase == "TRANSPORT":
            if not is_grasped: return "RELEASE"
        elif self.current_phase == "RELEASE":
            if dist > 0.2: return "RETREAT"
        elif self.current_phase == "RETREAT":
            if dist > 0.5: return "IDLE"
            
        return self.current_phase

    def _extract_state_features(self, obs, info) -> Dict[str, Any]:
        """Maps simulation observation/info to VidReward state dictionary."""
        # Note: Adroit obs contains qpos, qvel, and object info
        return {
            "qpos": obs[:28] if len(obs) >= 28 else np.zeros(28),
            "qvel": obs[28:56] if len(obs) >= 56 else np.zeros(28),
            # Add other features as needed by primitives
        }

    def set_locked_phase(self, phase: Optional[str]):
        self.locked_phase = phase

    def reset(self, **kwargs):
        self.current_phase = "IDLE"
        return self.env.reset(**kwargs)
