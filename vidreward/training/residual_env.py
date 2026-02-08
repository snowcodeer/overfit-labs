"""
Residual RL Environment Wrapper.

Implements the core residual RL paradigm from Johannink et al. (2018):
    π_final(s) = π_base(s) + α * π_residual(s)

The base policy comes from BC on video demonstration.
The residual policy is learned via RL to correct errors.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple

from vidreward.training.bc_policy import (
    BCBasePolicy,
    extract_relative_features,
    extract_features_from_sim_state
)


class ResidualRLEnv(gym.Wrapper):
    """
    Wraps an Adroit environment to implement residual RL.

    The RL agent outputs residual actions that are added to the base policy:
        action = base_action + residual_scale * residual_action

    Key features:
    - Base policy provides demonstration-derived actions
    - Residual actions are bounded and scaled
    - Residual penalty encourages staying close to demonstration
    - Supports curriculum learning via residual_scale adjustment
    """

    def __init__(
        self,
        env: gym.Env,
        base_policy: BCBasePolicy,
        residual_scale: float = 0.3,
        residual_penalty_weight: float = 0.1,
        use_delta_actions: bool = True,
        include_base_action_in_obs: bool = True,
    ):
        """
        Args:
            env: Base Adroit environment
            base_policy: Trained BC policy that outputs delta actions
            residual_scale: Scale factor for residual actions (curriculum param)
            residual_penalty_weight: Weight for penalizing large residuals
            use_delta_actions: If True, actions are deltas; else absolute targets
            include_base_action_in_obs: If True, append base action to observation
        """
        super().__init__(env)

        self.base_policy = base_policy
        self.residual_scale = residual_scale
        self.residual_penalty_weight = residual_penalty_weight
        self.use_delta_actions = use_delta_actions
        self.include_base_action_in_obs = include_base_action_in_obs

        # Store last base action for reward computation
        self._last_base_action = None
        self._last_residual_action = None

        # Modify observation space if including base action
        if include_base_action_in_obs:
            orig_obs_space = env.observation_space
            base_action_dim = base_policy.action_dim

            low = np.concatenate([
                orig_obs_space.low,
                np.full(base_action_dim, -np.inf)
            ])
            high = np.concatenate([
                orig_obs_space.high,
                np.full(base_action_dim, np.inf)
            ])
            self.observation_space = spaces.Box(
                low=low.astype(np.float32),
                high=high.astype(np.float32),
                dtype=np.float32
            )

        # Action space remains the same (residual in same space as base)
        # But we could optionally shrink it to reduce residual authority
        self.action_space = env.action_space

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and base policy state."""
        obs, info = self.env.reset(**kwargs)

        self._last_base_action = None
        self._last_residual_action = None

        # Get base action for augmented observation
        if self.include_base_action_in_obs:
            state_features = self._get_state_features()
            base_action = self.base_policy.get_action(state_features)
            obs = np.concatenate([obs, base_action])
            self._last_base_action = base_action

        return obs, info

    def step(self, residual_action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute composed action: base + scaled residual.

        Args:
            residual_action: Action from RL policy (in action space bounds)

        Returns:
            Standard gym step outputs with modified reward
        """
        # 1. Get current state features for base policy
        state_features = self._get_state_features()

        # 2. Get base action from BC policy
        base_action = self.base_policy.get_action(state_features)
        self._last_base_action = base_action

        # 3. Scale and clip residual
        scaled_residual = np.clip(
            residual_action * self.residual_scale,
            -self.residual_scale,
            self.residual_scale
        )
        self._last_residual_action = scaled_residual

        # 4. Compose final action
        if self.use_delta_actions:
            # Both base and residual are deltas
            current_qpos = self.env.unwrapped.data.qpos[:30].copy()
            target_qpos = current_qpos + base_action + scaled_residual
            final_action = np.clip(
                target_qpos,
                self.env.action_space.low,
                self.env.action_space.high
            )
        else:
            # Actions are absolute targets
            final_action = np.clip(
                base_action + scaled_residual,
                self.env.action_space.low,
                self.env.action_space.high
            )

        # 5. Step environment with composed action
        obs, env_reward, terminated, truncated, info = self.env.step(final_action)

        # 6. Compute residual penalty
        residual_penalty = -self.residual_penalty_weight * np.sum(scaled_residual ** 2)

        # 7. Combine rewards
        total_reward = env_reward + residual_penalty

        # 8. Add info for logging
        info['base_action_norm'] = np.linalg.norm(base_action)
        info['residual_action_norm'] = np.linalg.norm(scaled_residual)
        info['residual_penalty'] = residual_penalty
        info['env_reward'] = env_reward
        info['residual_scale'] = self.residual_scale

        # 9. Augment observation with base action if configured
        if self.include_base_action_in_obs:
            # Get base action for NEXT state (for next step's decision)
            next_state_features = self._get_state_features()
            next_base_action = self.base_policy.get_action(next_state_features)
            obs = np.concatenate([obs, next_base_action])

        return obs, total_reward, terminated, truncated, info

    def _get_state_features(self) -> np.ndarray:
        """Extract relative features from current sim state."""
        model = self.env.unwrapped.model
        data = self.env.unwrapped.data
        return extract_features_from_sim_state(model, data)

    def set_residual_scale(self, scale: float):
        """Update residual scale (for curriculum learning)."""
        self.residual_scale = scale

    def set_residual_penalty_weight(self, weight: float):
        """Update residual penalty weight."""
        self.residual_penalty_weight = weight

    def get_base_action(self) -> Optional[np.ndarray]:
        """Get the last computed base action."""
        return self._last_base_action

    def get_residual_action(self) -> Optional[np.ndarray]:
        """Get the last residual action (after scaling)."""
        return self._last_residual_action


class ResidualCurriculumCallback:
    """
    Callback for curriculum learning on residual scale.

    Gradually increases residual authority over training:
    - Early training: small residual scale (follow demo closely)
    - Late training: larger scale (allow corrections)
    """

    def __init__(
        self,
        env: ResidualRLEnv,
        initial_scale: float = 0.1,
        final_scale: float = 0.5,
        warmup_steps: int = 100_000,
        penalty_decay_rate: float = 0.99999,
        min_penalty_weight: float = 0.01,
    ):
        """
        Args:
            env: ResidualRLEnv instance
            initial_scale: Starting residual scale
            final_scale: Maximum residual scale
            warmup_steps: Steps to reach final scale
            penalty_decay_rate: Per-step decay for penalty weight
            min_penalty_weight: Minimum penalty weight
        """
        self.env = env
        self.initial_scale = initial_scale
        self.final_scale = final_scale
        self.warmup_steps = warmup_steps
        self.penalty_decay_rate = penalty_decay_rate
        self.min_penalty_weight = min_penalty_weight

        self.num_timesteps = 0
        self.initial_penalty_weight = env.residual_penalty_weight

    def on_step(self) -> Dict[str, float]:
        """
        Called after each environment step.

        Returns:
            Dictionary of curriculum metrics for logging
        """
        self.num_timesteps += 1

        # Linear warmup for residual scale
        progress = min(1.0, self.num_timesteps / self.warmup_steps)
        new_scale = self.initial_scale + progress * (self.final_scale - self.initial_scale)
        self.env.set_residual_scale(new_scale)

        # Exponential decay for penalty weight
        new_penalty = max(
            self.min_penalty_weight,
            self.initial_penalty_weight * (self.penalty_decay_rate ** self.num_timesteps)
        )
        self.env.set_residual_penalty_weight(new_penalty)

        return {
            'curriculum/residual_scale': new_scale,
            'curriculum/penalty_weight': new_penalty,
            'curriculum/progress': progress,
        }

    def reset(self):
        """Reset curriculum state."""
        self.num_timesteps = 0
        self.env.set_residual_scale(self.initial_scale)
        self.env.set_residual_penalty_weight(self.initial_penalty_weight)


class SB3ResidualCurriculumCallback:
    """
    Stable-Baselines3 compatible callback for residual curriculum.
    """

    def __init__(
        self,
        env: ResidualRLEnv,
        initial_scale: float = 0.1,
        final_scale: float = 0.5,
        warmup_steps: int = 100_000,
        penalty_decay_rate: float = 0.99999,
        min_penalty_weight: float = 0.01,
        verbose: int = 0,
    ):
        # Import here to avoid hard dependency
        from stable_baselines3.common.callbacks import BaseCallback
        self._base_class = BaseCallback

        self.env = env
        self.initial_scale = initial_scale
        self.final_scale = final_scale
        self.warmup_steps = warmup_steps
        self.penalty_decay_rate = penalty_decay_rate
        self.min_penalty_weight = min_penalty_weight
        self.verbose = verbose

        self.initial_penalty_weight = env.residual_penalty_weight

    def _on_step(self) -> bool:
        """Called after each step in training."""
        # Linear warmup for residual scale
        progress = min(1.0, self.num_timesteps / self.warmup_steps)
        new_scale = self.initial_scale + progress * (self.final_scale - self.initial_scale)
        self.env.set_residual_scale(new_scale)

        # Exponential decay for penalty weight
        new_penalty = max(
            self.min_penalty_weight,
            self.initial_penalty_weight * (self.penalty_decay_rate ** self.num_timesteps)
        )
        self.env.set_residual_penalty_weight(new_penalty)

        # Log to tensorboard
        if self.num_timesteps % 1000 == 0:
            self.logger.record('curriculum/residual_scale', new_scale)
            self.logger.record('curriculum/penalty_weight', new_penalty)

        return True


def make_residual_env(
    base_env_id: str = "AdroitHandRelocate-v1",
    base_policy: BCBasePolicy = None,
    residual_scale: float = 0.3,
    residual_penalty_weight: float = 0.1,
    **env_kwargs
) -> ResidualRLEnv:
    """
    Factory function to create a residual RL environment.

    Args:
        base_env_id: Gym environment ID
        base_policy: Trained BC base policy
        residual_scale: Initial residual scale
        residual_penalty_weight: Penalty for large residuals
        **env_kwargs: Additional arguments for base env

    Returns:
        ResidualRLEnv instance
    """
    base_env = gym.make(base_env_id, **env_kwargs)

    if base_policy is None:
        raise ValueError("base_policy is required. Train a BC policy first.")

    return ResidualRLEnv(
        env=base_env,
        base_policy=base_policy,
        residual_scale=residual_scale,
        residual_penalty_weight=residual_penalty_weight,
    )


# Testing
def test_residual_env():
    """Quick test of ResidualRLEnv."""
    import gymnasium as gym
    from vidreward.training.bc_policy import BCBasePolicy, BCTrainer

    print("Creating base environment...")
    base_env = gym.make("AdroitHandRelocate-v1", render_mode="rgb_array")

    # Create dummy BC policy for testing
    print("Creating dummy BC policy...")
    state_dim = 27  # hand_to_obj(3) + finger_joints(24)
    action_dim = 30

    # Generate dummy data
    T = 50
    dummy_states = np.random.randn(T, state_dim) * 0.1
    dummy_actions = np.random.randn(T, action_dim) * 0.01

    # Train minimal BC policy
    from vidreward.training.bc_policy import BCConfig
    config = BCConfig(epochs=10, augment_samples_per_frame=5)
    trainer = BCTrainer(config)
    bc_policy = trainer.train(dummy_states, dummy_actions, verbose=False)

    # Create residual env
    print("Creating ResidualRLEnv...")
    env = ResidualRLEnv(
        env=base_env,
        base_policy=bc_policy,
        residual_scale=0.2,
        residual_penalty_weight=0.1
    )

    # Test rollout
    print("Testing rollout...")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")

    total_reward = 0
    for step in range(10):
        # Random residual action
        residual_action = env.action_space.sample() * 0.1
        obs, reward, terminated, truncated, info = env.step(residual_action)

        total_reward += reward
        print(f"Step {step}: reward={reward:.4f}, "
              f"base_norm={info['base_action_norm']:.4f}, "
              f"residual_norm={info['residual_action_norm']:.4f}")

        if terminated or truncated:
            break

    print(f"Total reward: {total_reward:.4f}")
    env.close()
    print("ResidualRLEnv test complete!")


if __name__ == "__main__":
    test_residual_env()
