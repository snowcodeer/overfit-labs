data = self.env.unwrapped.data
model = self.env.unwrapped.model

obj_pos = data.xpos[model.body("Object").id].copy()
dist_to_target = np.linalg.norm(obj_pos - self.target_pos)

success = dist_to_target < 0.1
is_lifted = obj_pos[2] > 0.08
if is_lifted:
    self.ever_lifted = True
    
n_contacts = data.ncon

# 0. Base Time Penalty
reward = -0.1

# 1. Contact guidance 
reward += min(n_contacts, 5) * 0.1

# 2. Lift & Stability Bonus
if is_lifted:
    reward += 5.0 
    dist_reward = (1.0 - np.clip(dist_to_target, 0, 1)) * 10.0
    reward += dist_reward
    
# 3. Smooth Success Nearness
if dist_to_target < 0.2:
    reward += (0.2 - dist_to_target) * 50.0 

# 4. Final Success
if success:
    reward += 500.0
    
# 5. Delayed Drop Penalty 
if self.ever_lifted and not is_lifted:
    reward -= 20.0 
