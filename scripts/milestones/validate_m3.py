import numpy as np
import os
import cv2
from vidreward.rewards.pipeline import create_default_reward_function

def validate_m3(video_path: str):
    print(f"Starting M3 Validation for {video_path}...")
    
    # Get actual frame count from video
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if num_frames == 0:
        print("Error: Could not read frame count from video.")
        return

    # Mock object centroids for the video
    obj_centroids = np.zeros((num_frames, 2)) + 0.5
    
    # 1. Create the reward function from video
    print("Generating reward function from video...")
    reward_fn = create_default_reward_function(video_path, obj_centroids)
    
    # 2. Simulate a few RL steps
    print("Simulating RL steps...")
    for step in range(10):
        # Mock state and info
        state = {
            "qpos": np.zeros(28),
            "qvel": np.zeros(28)
        }
        info = {
            "hand_to_obj_dist": 0.5 - 0.04 * step,
            "finger_spread": 0.1,
            "obj_to_target_dist": 0.5,
            "is_grasped": step > 8
        }
        
        reward = reward_fn(state, info, step)
        print(f"Step {step} | Reward: {reward:.4f}")
    
    print("M3 Validation complete!")

if __name__ == "__main__":
    video = "data/pick-rubiks-cube.mp4"
    if os.path.exists(video):
        validate_m3(video)
    else:
        print(f"Video {video} not found. Please provide a valid video path.")
