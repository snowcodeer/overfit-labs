from stable_baselines3.common.callbacks import BaseCallback

class PhaseLockCallback(BaseCallback):
    """
    Callback that updates the VidRewardWrapper's locked_phase over time.
    Phase-locking curriculum:
    - 0-100k steps: Lock to APPROACH
    - 100k-300k steps: Lock to GRASP
    - 300k+ steps: Unlock (Dynamic phases)
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.milestones = [
            (100000, "APPROACH"),
            (300000, "GRASP"),
            (float('inf'), None)
        ]

    def _on_step(self) -> bool:
        # Update locked phase in the environment
        # Note: training_env might be a VecEnv, we need to access the underlying wrappers
        current_phase = None
        for threshold, phase in self.milestones:
            if self.num_timesteps < threshold:
                current_phase = phase
                break
        
        # Update all vectorized envs
        self.training_env.env_method("set_locked_phase", current_phase)
        
        if self.verbose > 0 and self.n_calls % 1000 == 0:
            print(f"Curriculum Update: Step {self.num_timesteps}, Locked Phase: {current_phase}")
            
        return True
