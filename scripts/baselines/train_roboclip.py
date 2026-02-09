"""
RoboCLIP Baseline Training Script

Trains an RL agent using dense visual rewards from a pre-trained Vision-Language Model (CLIP).
Reward = Cosine Similarity(CLIP(Current View), CLIP(Text Goal))

This evaluates if semantic understanding alone (without phase-based decomposition) is sufficient.
"""

import argparse
import os
import sys
import numpy as np
import torch
import open_clip
import gymnasium as gym
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.train_grasp_residual import extract_grasp_pose, GraspResidualEnv, PlotCallback

class RoboCLIPRewardWrapper(gym.Wrapper):
    """
    Computes reward based on CLIP similarity between current view and goal text.
    """
    def __init__(self, env, goal_text, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(env)
        self.device = device
        self.goal_text = goal_text
        
        print(f"Loading CLIP (ViT-B-32) on {device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.model.to(device)
        self.model.eval()
        
        # Pre-compute text embedding
        with torch.no_grad():
            text = self.tokenizer([goal_text]).to(device)
            self.text_features = self.model.encode_text(text)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
            
        print(f"RoboCLIP Goal: '{goal_text}'")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 1. Render for CLIP
        # Adroit rendering returns high-res, we resize for speed
        try:
            rgb = self.env.mujoco_renderer.render("rgb_array") # Direct access if possible
        except:
             rgb = self.env.render() # Standard API
             
        if rgb is None:
            # Fallback for some envs
            clip_reward = 0.0
        else:
            # 2. Preprocess & Encode
            image = Image.fromarray(rgb)
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # 3. Compute Similarity
                similarity = (image_features @ self.text_features.T).item()
                
            # Scale reward to be meaningful (e.g. 0.2 similarity -> 0.2 reward)
            # Center around neutral similarity? e.g. (sim - 0.2) * 10
            clip_reward = similarity
            
        # Combine: We replace the dense reward with CLIP reward, 
        # but keep sparse success reward if available (though RoboCLIP usually implies NO shaped reward)
        # For baseline comparison, we usually add it to sparse task reward
        
        # Let's say Total Reward = CLIP Similarity * 10.0 (Scale needed for PPO)
        # We ignore the environment's original reward to test pure CLIP guidance
        
        total_reward = clip_reward
        
        # Add success bonus if available (ground truth)
        if info.get('success', False):
            total_reward += 10.0
            
        info['clip_reward'] = clip_reward
        
        return obs, total_reward, terminated, truncated, info

def train(args):
    print(f"Training RoboCLIP Baseline for {args.video}")
    
    # 1. Load Task Config
    grasp_qpos, target_pos, _, _, grasp_frame, _, task_type = extract_grasp_pose(args.video)
    
    # Define Goal Text based on task
    if args.goal_text:
        goal_text = args.goal_text
    else:
        # Auto-generate naive prompts
        if task_type == 'catch':
            goal_text = "a robot hand catching a flying object"
        else:
            goal_text = "a robot hand picking up an object"
            
    def make_env(rank: int):
        def _init():
            # Enable rendering for CLIP
            base_env = gym.make("AdroitHandRelocate-v1", render_mode="rgb_array")
            
            # Use GraspResidualEnv for consistent initialization/termination logic
            # but NO trajectory, NO phases. Just physics.
            env = GraspResidualEnv(
                base_env,
                grasp_qpos=grasp_qpos,
                target_pos=target_pos,
                transport_traj=None, 
                grasp_frame_idx=grasp_frame,
                max_steps=args.max_steps,
                residual_scale=0.1,
                task_type=task_type,
                no_phases=True # Disable internal phase rewards
            )
            
            # Wrap with RoboCLIP
            env = RoboCLIPRewardWrapper(env, goal_text=goal_text)
            
            # Monitor
            monitor_path = os.path.join(args.output_dir, str(rank))
            env = Monitor(env, filename=monitor_path, info_keywords=('success', 'clip_reward'))
            return env
        return _init

    # 2. Create Envs
    # CLIP encoding is heavy, run fewer parallel envs? 
    # Or use Threaded? Subproc is safer for GPU usage if carefully managed.
    # OpenCLIP on GPU + PPO on GPU might conflict in subprocesses if not using 'forkserver' or 'spawn'.
    # For simplicity, let's use DummyVecEnv (Sequential) if num_envs is small, 
    # or ensure multiprocessing start method is compatible.
    # Windows uses 'spawn' by default, which is good.
    
    env_fns = [make_env(i) for i in range(args.num_envs)]
    if args.num_envs > 1:
        # On Windows, SubprocVecEnv works but requires if __name__ == '__main__' protection (we have it)
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)
        
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # 3. Train
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512, # Shorter rollout for heavier env?
        batch_size=64,
        tensorboard_log=os.path.join(args.output_dir, "tensorboard"),
        device="auto"
    )
    
    callbacks = [
        CheckpointCallback(save_freq=10000, save_path=os.path.join(args.output_dir, "checkpoints")),
         # reuse PlotCallback but it depends on 'lifted' which might not be in info if we use no_phases
         # Let's trust Monitor logs
    ]
    
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)
    model.save(os.path.join(args.output_dir, "final_model"))
    env.save(os.path.join(args.output_dir, "vec_normalize.pkl"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", default="runs/baselines/roboclip")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=1) # Default to 1 to save GPU/CPU memory
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--goal-text", type=str, default="")
    
    args = parser.parse_args()
    train(args)
