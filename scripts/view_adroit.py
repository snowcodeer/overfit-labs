import gymnasium as gym
import gymnasium_robotics
import numpy as np
import cv2
import os
from vidreward.training.env_wrapper import VidRewardWrapper
from vidreward.utils.video_io import VideoWriter

def view_adroit(output_path: str = "adroit_simulation.mp4", num_steps: int = 100):
    print(f"Initializing AdroitHandRelocate-v1...")
    # Use rgb_array for headless video capture
    env = gym.make("AdroitHandRelocate-v1", render_mode="rgb_array")
    
    # Reset to get dimensions
    obs, info = env.reset()
    frame_rgb = env.render()
    h, w, _ = frame_rgb.shape
    fps = 30
    
    # Wrap with VidReward AFTER getting initial frame to ensure clean render
    env = VidRewardWrapper(env)
    
    print(f"Recording simulation to {output_path} ({w}x{h})...")
    
    with VideoWriter(output_path, fps, int(w), int(h)) as writer:
        for i in range(num_steps):
            # Use random actions for now to see the scene
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            frame_rgb = env.render()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame = np.ascontiguousarray(frame)
            
            # Overlay info
            cv2.putText(frame, f"Step: {i}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Phase: {info.get('current_phase', 'N/A')}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Reward: {reward:.4f}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            writer.write_frame(frame)
            
            if terminated or truncated:
                obs, info = env.reset()
                print(f"Episode finished at step {i}")
    
    env.close()
    print(f"Done! Saved visualization to {output_path}")

if __name__ == "__main__":
    view_adroit()
