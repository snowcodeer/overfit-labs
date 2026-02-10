data = self.env.unwrapped.data
model = self.env.unwrapped.model

obj_pos = data.xpos[model.body("Object").id].copy()
dist_to_target = np.linalg.norm(obj_pos - self.target_pos)

success = dist_to_target < 0.1
is_lifted = obj_pos[2] > 0.08
n_contacts = data.ncon

reward = -0.1
reward += min(n_contacts, 5) * 0.1
if is_lifted:
    reward += 2.0
    dist_reward = (1.0 - np.clip(dist_to_target, 0, 1)) * 5.0
    reward += dist_reward
if success:
    reward += 50.0 # First success boost
if obj_pos[2] < 0.02: 
    reward -= 5.0
