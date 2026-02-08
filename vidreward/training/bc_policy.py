"""
Behavioral Cloning Base Policy for One-Shot Residual RL.

This module provides:
1. Feature extraction (relative, task-centric features)
2. Data augmentation for single-demo learning
3. BC policy network (small MLP)
4. Training loop with early stopping
5. Data collection via simulation replay
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import gymnasium as gym


@dataclass
class BCConfig:
    """Configuration for BC policy training."""
    # Network
    hidden_dims: Tuple[int, ...] = (128, 128)

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 1000
    early_stop_patience: int = 50

    # Augmentation
    augment_samples_per_frame: int = 50
    position_noise_std: float = 0.02  # 2cm
    joint_noise_std: float = 0.05     # ~3 degrees
    temporal_blend: bool = True


def extract_relative_features(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Extract task-centric relative features from Adroit observation.

    Features:
    - hand_to_obj: (3,) relative position from palm to object
    - finger_joints: (24,) current finger joint angles

    Total: 27D (minimal version)

    For full version, add:
    - obj_to_target: (3,) object to goal position
    - finger_vels: (24,) finger joint velocities
    Total: 54D
    """
    # Handle both dict obs and flat array obs
    if isinstance(obs, dict):
        palm_pos = obs.get('palm_pos', obs.get('hand_pos', np.zeros(3)))
        obj_pos = obs.get('obj_pos', np.zeros(3))
        qpos = obs.get('qpos', np.zeros(30))
    else:
        # Flat observation from AdroitHandRelocate-v1
        # obs layout: qpos(30) + qvel(30) + palm_pos(3) + obj_pos(3) + target_pos(3)
        # Total: 69 or similar - need to check actual env
        qpos = obs[:30] if len(obs) >= 30 else np.zeros(30)
        # Approximate palm_pos from qpos (wrist position)
        palm_pos = qpos[:3]  # Base position
        obj_pos = obs[63:66] if len(obs) >= 66 else np.zeros(3)

    # Relative hand-to-object position
    hand_to_obj = obj_pos - palm_pos

    # Finger joint angles (skip wrist/base: indices 6-29 are fingers)
    finger_joints = qpos[6:30]

    return np.concatenate([hand_to_obj, finger_joints])


def extract_features_from_sim_state(
    model,
    data,
    include_velocity: bool = False
) -> np.ndarray:
    """
    Extract relative features directly from MuJoCo model/data.
    More accurate than observation-based extraction.
    """
    # Get palm position from body xpos
    try:
        palm_body_id = model.body("palm").id
        palm_pos = data.xpos[palm_body_id].copy()
    except:
        palm_pos = data.qpos[:3].copy()

    # Get object position
    try:
        obj_body_id = model.body("Object").id
        obj_pos = data.xpos[obj_body_id].copy()
    except:
        obj_pos = np.zeros(3)

    # Relative position
    hand_to_obj = obj_pos - palm_pos

    # Finger joints (indices 6-29 in qpos)
    finger_joints = data.qpos[6:30].copy()

    features = [hand_to_obj, finger_joints]

    if include_velocity:
        finger_vels = data.qvel[6:30].copy()
        features.append(finger_vels)

    return np.concatenate(features)


class SingleDemoAugmenter:
    """
    Data augmentation for single-demonstration BC training.

    Key insight: When we add noise to the state, we keep the original action.
    This teaches the policy to "correct" back toward the demonstration trajectory.
    """

    def __init__(
        self,
        position_noise_std: float = 0.02,
        joint_noise_std: float = 0.05,
        temporal_blend: bool = True
    ):
        self.position_noise_std = position_noise_std
        self.joint_noise_std = joint_noise_std
        self.temporal_blend = temporal_blend

    def augment(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        num_samples_per_frame: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment state-action pairs.

        Args:
            states: (T, state_dim) original states
            actions: (T, action_dim) original delta actions
            num_samples_per_frame: augmentation multiplier

        Returns:
            augmented_states: (T * (1 + num_samples), state_dim)
            augmented_actions: (T * (1 + num_samples), action_dim)
        """
        T, state_dim = states.shape
        action_dim = actions.shape[1]

        aug_states = [states.copy()]  # Include original
        aug_actions = [actions.copy()]

        for _ in range(num_samples_per_frame):
            noisy_states = states.copy()

            # 1. Position noise (first 3 dims are hand_to_obj)
            noisy_states[:, :3] += np.random.normal(
                0, self.position_noise_std, (T, 3)
            )

            # 2. Joint angle noise (dims 3-27 are finger joints)
            noisy_states[:, 3:] += np.random.normal(
                0, self.joint_noise_std, (T, state_dim - 3)
            )

            # Action stays the same - teaches correction back to trajectory
            aug_states.append(noisy_states)
            aug_actions.append(actions.copy())

        # 3. Temporal blending (interpolate between consecutive frames)
        if self.temporal_blend and T > 1:
            for _ in range(num_samples_per_frame // 2):
                alpha = np.random.uniform(0, 1, T - 1)

                blended_states = np.zeros((T - 1, state_dim))
                blended_actions = np.zeros((T - 1, action_dim))

                for t in range(T - 1):
                    a = alpha[t]
                    blended_states[t] = (1 - a) * states[t] + a * states[t + 1]
                    blended_actions[t] = (1 - a) * actions[t] + a * actions[t + 1]

                aug_states.append(blended_states)
                aug_actions.append(blended_actions)

        return np.vstack(aug_states), np.vstack(aug_actions)


class BCBasePolicy(nn.Module):
    """
    Behavioral Cloning policy network.

    Maps relative state features to delta joint actions.
    Intentionally small to prevent overfitting on single demo.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 128)
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build network
        layers = []
        prev_dim = state_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
            ])
            prev_dim = h

        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())  # Bound outputs to [-1, 1]

        self.net = nn.Sequential(*layers)

        # Normalization stats (set during training)
        self.register_buffer('state_mean', torch.zeros(state_dim))
        self.register_buffer('state_std', torch.ones(state_dim))
        self.register_buffer('action_scale', torch.ones(action_dim))

        # Initialize with small weights for stable start
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_normalization(
        self,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        action_scale: np.ndarray
    ):
        """Set normalization statistics from training data."""
        self.state_mean = torch.FloatTensor(state_mean)
        self.state_std = torch.FloatTensor(state_std)
        self.action_scale = torch.FloatTensor(action_scale)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass with normalization."""
        state_norm = (state - self.state_mean) / (self.state_std + 1e-8)
        action_norm = self.net(state_norm)
        return action_norm * self.action_scale

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Get action for a single state (numpy interface)."""
        self.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action_t = self.forward(state_t)
            return action_t.squeeze(0).numpy()

    def get_action_batch(self, states: np.ndarray) -> np.ndarray:
        """Get actions for batch of states."""
        self.eval()
        with torch.no_grad():
            states_t = torch.FloatTensor(states)
            actions_t = self.forward(states_t)
            return actions_t.numpy()


class BCTrainer:
    """
    Trainer for BC policy with early stopping and validation.
    """

    def __init__(self, config: BCConfig = None):
        self.config = config or BCConfig()
        self.augmenter = SingleDemoAugmenter(
            position_noise_std=self.config.position_noise_std,
            joint_noise_std=self.config.joint_noise_std,
            temporal_blend=self.config.temporal_blend
        )

    def collect_data_from_sim(
        self,
        env: gym.Env,
        joint_trajectory: np.ndarray,
        include_velocity: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect state-action pairs by replaying trajectory in simulation.

        This is more accurate than computing features from the trajectory
        directly, because we get ground-truth relative positions from sim.

        Args:
            env: Adroit environment
            joint_trajectory: (T, 30) joint angles from retargeting
            include_velocity: whether to include velocity in features

        Returns:
            states: (T-1, state_dim) relative features
            actions: (T-1, 30) delta joint angles
        """
        import mujoco

        model = env.unwrapped.model
        data = env.unwrapped.data

        T = len(joint_trajectory)
        states = []
        actions = []

        env.reset()

        for t in range(T - 1):
            # Set sim to current trajectory pose
            data.qpos[:30] = joint_trajectory[t]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)

            # Extract relative features
            state = extract_features_from_sim_state(
                model, data, include_velocity=include_velocity
            )
            states.append(state)

            # Delta action to next frame
            action = joint_trajectory[t + 1] - joint_trajectory[t]
            actions.append(action)

        return np.array(states), np.array(actions)

    def train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        val_split: float = 0.1,
        verbose: bool = True
    ) -> BCBasePolicy:
        """
        Train BC policy on state-action pairs.

        Args:
            states: (N, state_dim) training states
            actions: (N, action_dim) training actions (deltas)
            val_split: fraction for validation
            verbose: print training progress

        Returns:
            Trained BCBasePolicy
        """
        # Augment data
        if verbose:
            print(f"Original data: {len(states)} samples")

        aug_states, aug_actions = self.augmenter.augment(
            states, actions,
            num_samples_per_frame=self.config.augment_samples_per_frame
        )

        if verbose:
            print(f"After augmentation: {len(aug_states)} samples")

        # Split train/val
        n = len(aug_states)
        n_val = int(n * val_split)
        indices = np.random.permutation(n)

        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        train_states = aug_states[train_idx]
        train_actions = aug_actions[train_idx]
        val_states = aug_states[val_idx]
        val_actions = aug_actions[val_idx]

        # Compute normalization stats from training data
        state_mean = train_states.mean(axis=0)
        state_std = train_states.std(axis=0) + 1e-8
        action_scale = np.abs(train_actions).max(axis=0) + 1e-8

        # Create policy
        state_dim = states.shape[1]
        action_dim = actions.shape[1]

        policy = BCBasePolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.config.hidden_dims
        )
        policy.set_normalization(state_mean, state_std, action_scale)

        # Normalize data for training
        train_states_norm = (train_states - state_mean) / state_std
        train_actions_norm = train_actions / action_scale
        val_states_norm = (val_states - state_mean) / state_std
        val_actions_norm = val_actions / action_scale

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(train_states_norm),
            torch.FloatTensor(train_actions_norm)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        val_states_t = torch.FloatTensor(val_states_norm)
        val_actions_t = torch.FloatTensor(val_actions_norm)

        # Optimizer
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_state_dict = None

        for epoch in range(self.config.epochs):
            # Train
            policy.train()
            train_loss = 0.0
            for batch_states, batch_actions in train_loader:
                pred = policy.net(batch_states)
                loss = F.mse_loss(pred, batch_actions)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            policy.eval()
            with torch.no_grad():
                val_pred = policy.net(val_states_t)
                val_loss = F.mse_loss(val_pred, val_actions_t).item()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = {k: v.clone() for k, v in policy.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}: train_loss={train_loss:.6f}, "
                      f"val_loss={val_loss:.6f}, best={best_val_loss:.6f}")

            if patience_counter >= self.config.early_stop_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best weights
        if best_state_dict is not None:
            policy.load_state_dict(best_state_dict)

        if verbose:
            print(f"Training complete. Best val_loss: {best_val_loss:.6f}")

        return policy


def create_bc_base_policy_from_video(
    video_path: str,
    env: gym.Env,
    config: BCConfig = None,
    verbose: bool = True
) -> BCBasePolicy:
    """
    End-to-end: video -> BC base policy.

    Args:
        video_path: path to demonstration video
        env: Adroit environment for sim replay
        config: BC training configuration
        verbose: print progress

    Returns:
        Trained BCBasePolicy
    """
    from vidreward.extraction.mediapipe_tracker import MediaPipeTracker
    from vidreward.retargeting.landmarks_to_angles import AdroitRetargeter
    from vidreward.utils.video_io import VideoReader

    if verbose:
        print(f"Processing video: {video_path}")

    # 1. Extract hand trajectory
    reader = VideoReader(video_path)
    frames = list(reader.read_frames())

    tracker = MediaPipeTracker()
    hand_traj = tracker.process_frames(frames)
    tracker.close()

    if verbose:
        print(f"Extracted {len(hand_traj.landmarks)} frames of hand landmarks")

    # 2. Retarget to joint angles
    retargeter = AdroitRetargeter()
    joint_trajectory = retargeter.retarget_sequence(hand_traj.landmarks)

    # Fix NaN values
    joint_trajectory = np.nan_to_num(joint_trajectory, nan=0.0)

    if verbose:
        print(f"Retargeted to {joint_trajectory.shape} joint trajectory")

    # 3. Collect data via sim replay
    trainer = BCTrainer(config)
    states, actions = trainer.collect_data_from_sim(env, joint_trajectory)

    if verbose:
        print(f"Collected {len(states)} state-action pairs from sim")

    # 4. Train BC policy
    policy = trainer.train(states, actions, verbose=verbose)

    return policy


# Convenience function for quick testing
def test_bc_policy():
    """Quick test of BC policy on AdroitHandRelocate."""
    import gymnasium as gym

    # Create env
    env = gym.make("AdroitHandRelocate-v1", render_mode="rgb_array")

    # Generate dummy trajectory (for testing without video)
    T = 100
    dummy_joint_traj = np.zeros((T, 30))
    # Simple reaching motion
    for t in range(T):
        progress = t / T
        dummy_joint_traj[t, 0] = -0.1 + 0.2 * progress  # Move forward
        dummy_joint_traj[t, 2] = 0.2 - 0.1 * progress   # Move down
        # Close fingers gradually
        dummy_joint_traj[t, 8:30] = 0.5 * progress

    # Train
    config = BCConfig(epochs=100)  # Fewer epochs for testing
    trainer = BCTrainer(config)

    states, actions = trainer.collect_data_from_sim(env, dummy_joint_traj)
    print(f"State shape: {states.shape}, Action shape: {actions.shape}")

    policy = trainer.train(states, actions, verbose=True)

    # Test inference
    test_state = states[0]
    action = policy.get_action(test_state)
    print(f"Test action shape: {action.shape}")
    print(f"Test action: {action[:5]}...")  # First 5 dims

    env.close()
    print("BC policy test complete!")


if __name__ == "__main__":
    test_bc_policy()
