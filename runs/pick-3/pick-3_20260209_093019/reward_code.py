data = self.env.unwrapped.data
model = self.env.unwrapped.model

obj_pos = data.xpos[model.body("Object").id].copy()
dist_to_target = np.linalg.norm(obj_pos - self.target_pos)

success = dist_to_target < 0.1
is_lifted = obj_pos[2] > 0.08
n_contacts = data.ncon

# 0. Time penalty to encourage efficiency
reward = -0.1

# 1. Contact guidance (small bonus)
reward += min(n_contacts, 5) * 0.1

# 2. Lift and Transport
if is_lifted:
    reward += 2.0
    # Transport bonus: scaled up to encourage motion towards target
    # (1 - distance) * 10 => Max 10 per step at target
    dist_reward = (1.0 - np.clip(dist_to_target, 0, 1)) * 10.0
    reward += dist_reward

# 3. Success (Significant terminal bonus)
if success:
    reward += 500.0
    
# Penalty for dropping or going below table height
if obj_pos[2] < 0.02: 
    reward -= 5.0
