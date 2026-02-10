"""
Configuration for: pick-3
Task Type: slide
"""

TASK_NAME = "pick-3"
TASK_TYPE = "slide"
GRASP_FRAME = 0
RELEASE_FRAME = 100

# Milestones:
#   - approach: frame 0
#   - grasp: frame 45
#   - slide: frame 65
#   - release: frame 210
#   - retract: frame 240

MILESTONES = {
    "approach": 0,
    "grasp": 45,
    "slide": 65,
    "release": 210,
    "retract": 240
}

REWARD_CONFIG = {
    "time_penalty": -0.1,
    "contact_scale": 0.1,
    "success_bonus": 500.0,
    "approach_bonus": 10.0,
    "grasp_bonus": 10.0,
    "slide_bonus": 10.0,
    "release_bonus": 10.0,
    "retract_bonus": 10.0
}

BC_CONFIG = {
    "hidden_dims": (128, 128),
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 256,
    "epochs": 1000,
    "early_stop_patience": 50,
    "augment_samples_per_frame": 50,
    "position_noise_std": 0.02,
    "joint_noise_std": 0.05,
    "temporal_blend": True,
}

RESIDUAL_CONFIG = {
    "initial_scale": 0.1,
    "final_scale": 0.5,
    "warmup_steps": 100_000,
    "penalty_weight": 0.1,
    "penalty_decay_rate": 0.99999,
    "min_penalty_weight": 0.01,
    "use_delta_actions": True,
    "include_base_in_obs": True,
}

TRAINING_CONFIG = {
    "algorithm": "TD3",
    "total_timesteps": 50_000,
    "learning_rate": 3e-4,
    "buffer_size": 100_000,
    "learning_starts": 1000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "policy_arch": [256, 256],
    "action_noise_sigma": 0.1,
    "max_episode_steps": 100,
}
