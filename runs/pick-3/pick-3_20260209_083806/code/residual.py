"""
Residual RL Environment Wrapper for: pick-3
Task Type: slide

action = base_policy(s) + alpha * residual_policy(s)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Tuple

from config import RESIDUAL_CONFIG
from baseline import BCBasePolicy, extract_features_from_sim


class ResidualRLEnv(gym.Wrapper):
    """
    action = base_action + residual_scale * residual_action
    """

    def __init__(self, env, base_policy, residual_scale=0.1, residual_penalty_weight=0.1):
        super().__init__(env)
        self.base_policy = base_policy
        self.residual_scale = residual_scale
        self.residual_penalty_weight = residual_penalty_weight

    def step(self, residual_action):
        features = self._get_features()
        base_action = self.base_policy.get_action(features)

        scaled_residual = np.clip(
            residual_action * self.residual_scale,
            -self.residual_scale,
            self.residual_scale
        )

        current_qpos = self.env.unwrapped.data.qpos[:30].copy()
        final_action = np.clip(
            current_qpos + base_action + scaled_residual,
            self.env.action_space.low,
            self.env.action_space.high
        )

        obs, env_reward, terminated, truncated, info = self.env.step(final_action)
        residual_penalty = -self.residual_penalty_weight * np.sum(scaled_residual ** 2)

        return obs, env_reward + residual_penalty, terminated, truncated, info

    def _get_features(self):
        return extract_features_from_sim(self.env.unwrapped.model, self.env.unwrapped.data)

    def set_residual_scale(self, scale):
        self.residual_scale = scale


class ResidualCurriculum:
    """Curriculum learning for residual scale."""

    def __init__(self, env, initial_scale=0.1, final_scale=0.5, warmup_steps=100000):
        self.env = env
        self.initial_scale = initial_scale
        self.final_scale = final_scale
        self.warmup_steps = warmup_steps
        self.num_steps = 0

    def on_step(self):
        self.num_steps += 1
        progress = min(1.0, self.num_steps / self.warmup_steps)
        new_scale = self.initial_scale + progress * (self.final_scale - self.initial_scale)
        self.env.set_residual_scale(new_scale)
        return {"residual_scale": new_scale, "progress": progress}
