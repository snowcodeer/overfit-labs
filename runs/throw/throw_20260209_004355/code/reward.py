"""
Reward Function for: throw
Task Type: throw
"""

import numpy as np
from config import REWARD_CONFIG, MILESTONES


def compute_reward(env, obs, action, info):
    data = env.unwrapped.data
    model = env.unwrapped.model

    obj_pos = data.xpos[model.body("Object").id].copy()
    palm_pos = data.xpos[model.body("palm").id].copy()
    dist_to_target = np.linalg.norm(obj_pos - env.target_pos)
    n_contacts = data.ncon
    obj_height = obj_pos[2]

    reward = REWARD_CONFIG.get("time_penalty", -0.1)
    reward += min(n_contacts, 5) * REWARD_CONFIG.get("contact_scale", 0.1)

    completed = getattr(env, "_completed_milestones", set())

    for name, frame in MILESTONES.items():
        if name in completed:
            continue
        bonus = REWARD_CONFIG.get(f"{name}_bonus", 10.0)

        if "grasp" in name.lower() and n_contacts >= 3:
            reward += bonus
            completed.add(name)
        elif "lift" in name.lower() and obj_height > 0.1:
            reward += bonus
            completed.add(name)

    env._completed_milestones = completed

    success = dist_to_target < 0.1
    if success:
        reward += REWARD_CONFIG.get("success_bonus", 500.0)

    return reward, {"success": success, "obj_height": obj_height, "n_contacts": n_contacts}
