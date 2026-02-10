"""
Configuration for: IMG_0754
Task Type: slide
Auto-generated from video analysis.

This file contains ALL hyperparameters for the residual RL pipeline:
- Task configuration (from video analysis)
- Reward configuration (milestone bonuses)
- BC (Behavioral Cloning) configuration
- Residual RL configuration
- Training configuration
"""

# ============================================================================
# TASK CONFIGURATION (from video analysis)
# ============================================================================
TASK_NAME = "IMG_0754"
TASK_TYPE = "slide"
GRASP_FRAME = 0
RELEASE_FRAME = 100

# Milestones detected from video:
#   - approach: frame 0 (Hand begins moving towards the blue eraser.)
#   - contact: frame 18 (The fingers make first contact with the sides of the eraser.)
#   - grasp: frame 30 (The hand has a firm grip on the eraser, preparing to move it.)
#   - start_slide: frame 45 (The hand begins sliding the eraser across the gray surface.)
#   - stop_slide: frame 115 (The eraser reaches its new position and movement stops.)
#   - release: frame 128 (The fingers let go of the eraser.)
#   - retreat: frame 150 (The hand moves away from the object and out of the frame.)

MILESTONES = {
    "approach": 0,
    "contact": 18,
    "grasp": 30,
    "start_slide": 45,
    "stop_slide": 115,
    "release": 128,
    "retreat": 150
}


# ============================================================================
# REWARD CONFIGURATION - Edit these to tune reward shaping
# ============================================================================
REWARD_CONFIG = {
    "time_penalty": -0.1,
    "contact_scale": 0.1,
    "success_bonus": 500.0,
    "approach_bonus": 10.0,
    "contact_bonus": 10.0,
    "grasp_bonus": 10.0,
    "start_slide_bonus": 10.0,
    "stop_slide_bonus": 10.0,
    "release_bonus": 10.0,
    "retreat_bonus": 10.0
}


# ============================================================================
# BC (BEHAVIORAL CLONING) CONFIGURATION - Tune the base policy
# ============================================================================
BC_CONFIG = {
    # Network architecture
    "hidden_dims": (128, 128),      # MLP hidden layer sizes

    # Training
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 256,
    "epochs": 1000,
    "early_stop_patience": 50,

    # Data augmentation (critical for single-demo learning!)
    "augment_samples_per_frame": 50,  # Augmented samples per original frame
    "position_noise_std": 0.02,       # Position noise in meters (2cm)
    "joint_noise_std": 0.05,          # Joint angle noise in radians (~3 deg)
    "temporal_blend": True,           # Interpolate between consecutive frames
}


# ============================================================================
# RESIDUAL RL CONFIGURATION - Tune the residual learning
# ============================================================================
RESIDUAL_CONFIG = {
    # Residual scale curriculum (KEY hyperparameters!)
    # action = base_policy(s) + residual_scale * rl_policy(s)
    "initial_scale": 0.1,       # Start small - follow demo closely
    "final_scale": 0.5,         # End larger - allow bigger corrections
    "warmup_steps": 100_000,    # Steps to reach final scale

    # Residual penalty (keeps RL close to demo)
    "penalty_weight": 0.1,      # Weight for -||residual||^2 penalty
    "penalty_decay_rate": 0.99999,
    "min_penalty_weight": 0.01,

    # Action composition
    "use_delta_actions": True,  # True = delta actions, False = absolute
    "include_base_in_obs": True,  # Include base action in RL observation
}


# ============================================================================
# TRAINING CONFIGURATION - Overall training settings
# ============================================================================
TRAINING_CONFIG = {
    # Algorithm
    "algorithm": "TD3",
    "total_timesteps": 50_000,

    # TD3 hyperparameters
    "learning_rate": 3e-4,
    "buffer_size": 100_000,
    "learning_starts": 1000,
    "batch_size": 256,
    "tau": 0.005,               # Target network update rate
    "gamma": 0.99,              # Discount factor

    # Policy network
    "policy_arch": [256, 256],

    # Exploration noise
    "action_noise_sigma": 0.1,

    # Environment
    "max_episode_steps": 100,
}
