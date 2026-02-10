"""
Behavioral Cloning Base Policy for: IMG_0754
Task Type: slide

This module provides the BC base policy that the residual RL learns on top of.
The key insight: BC from a single demo is fragile, but it gets us close.
RL learns small corrections to fix the errors.

Pipeline: Video -> MediaPipe -> Retarget -> BC Policy -> Residual RL
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple
from dataclasses import dataclass

from config import BC_CONFIG


@dataclass
class BCConfig:
    """Configuration for BC policy training."""
    hidden_dims: Tuple[int, ...] = BC_CONFIG.get("hidden_dims", (128, 128))
    learning_rate: float = BC_CONFIG.get("learning_rate", 1e-3)
    weight_decay: float = BC_CONFIG.get("weight_decay", 1e-4)
    batch_size: int = BC_CONFIG.get("batch_size", 256)
    epochs: int = BC_CONFIG.get("epochs", 1000)
    early_stop_patience: int = BC_CONFIG.get("early_stop_patience", 50)
    augment_samples_per_frame: int = BC_CONFIG.get("augment_samples_per_frame", 50)
    position_noise_std: float = BC_CONFIG.get("position_noise_std", 0.02)
    joint_noise_std: float = BC_CONFIG.get("joint_noise_std", 0.05)
    temporal_blend: bool = BC_CONFIG.get("temporal_blend", True)


def extract_features_from_sim(model, data) -> np.ndarray:
    """
    Extract relative features from MuJoCo state.

    Features:
    - hand_to_obj: (3,) relative position from palm to object
    - finger_joints: (24,) current finger joint angles

    Total: 27D state representation
    """
    # Get palm position
    try:
        palm_pos = data.xpos[model.body("palm").id].copy()
    except:
        palm_pos = data.qpos[:3].copy()

    # Get object position
    try:
        obj_pos = data.xpos[model.body("Object").id].copy()
    except:
        obj_pos = np.zeros(3)

    # Relative position (task-centric feature)
    hand_to_obj = obj_pos - palm_pos

    # Finger joints (indices 6-29 in qpos)
    finger_joints = data.qpos[6:30].copy()

    return np.concatenate([hand_to_obj, finger_joints])


class SingleDemoAugmenter:
    """
    Data augmentation for single-demonstration BC training.

    Key insight: When we add noise to the state, we keep the original action.
    This teaches the policy to "correct" back toward the demonstration.
    """

    def __init__(self, config: BCConfig = None):
        config = config or BCConfig()
        self.position_noise_std = config.position_noise_std
        self.joint_noise_std = config.joint_noise_std
        self.temporal_blend = config.temporal_blend

    def augment(self, states: np.ndarray, actions: np.ndarray,
                num_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Augment state-action pairs for single-demo learning."""
        T, state_dim = states.shape

        aug_states = [states.copy()]
        aug_actions = [actions.copy()]

        for _ in range(num_samples):
            noisy_states = states.copy()

            # Position noise (first 3 dims = hand_to_obj)
            noisy_states[:, :3] += np.random.normal(0, self.position_noise_std, (T, 3))

            # Joint noise (dims 3-27 = finger joints)
            noisy_states[:, 3:] += np.random.normal(0, self.joint_noise_std, (T, state_dim - 3))

            # Action stays the same - teaches correction!
            aug_states.append(noisy_states)
            aug_actions.append(actions.copy())

        # Temporal blending
        if self.temporal_blend and T > 1:
            for _ in range(num_samples // 2):
                alpha = np.random.uniform(0, 1, T - 1)
                blended_states = np.zeros((T - 1, state_dim))
                blended_actions = np.zeros((T - 1, actions.shape[1]))

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

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: Tuple[int, ...] = (128, 128)):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build MLP
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

        # Normalization stats
        self.register_buffer('state_mean', torch.zeros(state_dim))
        self.register_buffer('state_std', torch.ones(state_dim))
        self.register_buffer('action_scale', torch.ones(action_dim))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_normalization(self, state_mean, state_std, action_scale):
        self.state_mean = torch.FloatTensor(state_mean)
        self.state_std = torch.FloatTensor(state_std)
        self.action_scale = torch.FloatTensor(action_scale)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
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


class BCTrainer:
    """Trainer for BC policy with augmentation and early stopping."""

    def __init__(self, config: BCConfig = None):
        self.config = config or BCConfig()
        self.augmenter = SingleDemoAugmenter(self.config)

    def collect_data_from_sim(self, env, joint_trajectory: np.ndarray):
        """
        Collect state-action pairs by replaying trajectory in simulation.

        Args:
            env: Adroit environment
            joint_trajectory: (T, 30) joint angles from retargeting

        Returns:
            states: (T-1, 27) relative features
            actions: (T-1, 30) delta joint angles
        """
        import mujoco

        model = env.unwrapped.model
        data = env.unwrapped.data

        T = len(joint_trajectory)
        states, actions = [], []

        env.reset()

        for t in range(T - 1):
            # Set sim to trajectory pose
            data.qpos[:30] = joint_trajectory[t]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)

            # Extract features
            state = extract_features_from_sim(model, data)
            states.append(state)

            # Delta action
            action = joint_trajectory[t + 1] - joint_trajectory[t]
            actions.append(action)

        return np.array(states), np.array(actions)

    def train(self, states: np.ndarray, actions: np.ndarray,
              verbose: bool = True) -> BCBasePolicy:
        """Train BC policy with augmentation and early stopping."""

        if verbose:
            print(f"Original data: {len(states)} samples")

        # Augment data
        aug_states, aug_actions = self.augmenter.augment(
            states, actions, self.config.augment_samples_per_frame
        )

        if verbose:
            print(f"After augmentation: {len(aug_states)} samples")

        # Train/val split
        n = len(aug_states)
        n_val = int(n * 0.1)
        indices = np.random.permutation(n)

        train_states = aug_states[indices[n_val:]]
        train_actions = aug_actions[indices[n_val:]]
        val_states = aug_states[indices[:n_val]]
        val_actions = aug_actions[indices[:n_val]]

        # Normalization
        state_mean = train_states.mean(axis=0)
        state_std = train_states.std(axis=0) + 1e-8
        action_scale = np.abs(train_actions).max(axis=0) + 1e-8

        # Create policy
        policy = BCBasePolicy(
            state_dim=states.shape[1],
            action_dim=actions.shape[1],
            hidden_dims=self.config.hidden_dims
        )
        policy.set_normalization(state_mean, state_std, action_scale)

        # Normalize data
        train_states_norm = (train_states - state_mean) / state_std
        train_actions_norm = train_actions / action_scale
        val_states_norm = (val_states - state_mean) / state_std
        val_actions_norm = val_actions / action_scale

        # DataLoader
        train_dataset = TensorDataset(
            torch.FloatTensor(train_states_norm),
            torch.FloatTensor(train_actions_norm)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

        val_states_t = torch.FloatTensor(val_states_norm)
        val_actions_t = torch.FloatTensor(val_actions_norm)

        # Optimizer
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_state_dict = None

        for epoch in range(self.config.epochs):
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
                print(f"Epoch {epoch + 1}: train={train_loss:.6f}, val={val_loss:.6f}")

            if patience_counter >= self.config.early_stop_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        if best_state_dict:
            policy.load_state_dict(best_state_dict)

        if verbose:
            print(f"BC training complete. Best val_loss: {best_val_loss:.6f}")

        return policy
