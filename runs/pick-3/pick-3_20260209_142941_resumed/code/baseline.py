"""
Behavioral Cloning Base Policy for: pick-3
Task Type: slide

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
    hidden_dims: Tuple[int, ...] = BC_CONFIG.get("hidden_dims", (128, 128))
    learning_rate: float = BC_CONFIG.get("learning_rate", 1e-3)
    batch_size: int = BC_CONFIG.get("batch_size", 256)
    epochs: int = BC_CONFIG.get("epochs", 1000)
    augment_samples_per_frame: int = BC_CONFIG.get("augment_samples_per_frame", 50)


def extract_features_from_sim(model, data) -> np.ndarray:
    """Extract relative features from MuJoCo state."""
    try:
        palm_pos = data.xpos[model.body("palm").id].copy()
    except:
        palm_pos = data.qpos[:3].copy()

    try:
        obj_pos = data.xpos[model.body("Object").id].copy()
    except:
        obj_pos = np.zeros(3)

    hand_to_obj = obj_pos - palm_pos
    finger_joints = data.qpos[6:30].copy()

    return np.concatenate([hand_to_obj, finger_joints])


class BCBasePolicy(nn.Module):
    """Behavioral Cloning policy network."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(128, 128)):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        layers = []
        prev_dim = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.LayerNorm(h), nn.ReLU()])
            prev_dim = h
        layers.extend([nn.Linear(prev_dim, action_dim), nn.Tanh()])

        self.net = nn.Sequential(*layers)
        self.register_buffer('state_mean', torch.zeros(state_dim))
        self.register_buffer('state_std', torch.ones(state_dim))
        self.register_buffer('action_scale', torch.ones(action_dim))

    def forward(self, state):
        state_norm = (state - self.state_mean) / (self.state_std + 1e-8)
        return self.net(state_norm) * self.action_scale

    def get_action(self, state: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            return self.forward(state_t).squeeze(0).numpy()
