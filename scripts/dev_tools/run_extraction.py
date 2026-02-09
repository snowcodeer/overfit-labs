import os
import cv2
import numpy as np
import argparse
from vidreward.extraction.mediapipe_tracker import MediaPipeTracker
from vidreward.extraction.sam2_tracker import SAM2Tracker
from vidreward.extraction.trajectory import HandTrajectory, ObjectTrajectory, VideoTrajectory
from vidreward.utils.video_io import VideoReader, VideoWriter
from vidreward.utils.visualization import draw_landmarks, draw_mask, plot_distances

def run_extraction(video_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(video_path).split('.')[0]
    
    print(f"Processing {video_path}...")
    
    # 1. Initialize Trackers
    mp_tracker = MediaPipeTracker()
    # SAM2 requires weights - providing placeholders for now
    # sam2_tracker = SAM2Tracker("checkpoints/sam2_hiera_large.pt", "sam2_hiera_l.yaml")
    
    reader = VideoReader(video_path)
    fps = reader.fps
    width, height = reader.width, reader.height
    
    hand_landmarks = []
    hand_confidences = []
    
    # Process for Hand Landmarks
    print("Extracting hand landmarks...")
    for frame in reader.read_frames():
        lms, conf = mp_tracker.process_frame(frame)
        if lms is not None:
            hand_landmarks.append(lms)
            hand_confidences.append(conf)
        else:
            hand_landmarks.append(np.zeros((21, 3)))
            hand_confidences.append(0.0)
            
    hand_traj = HandTrajectory(np.array(hand_landmarks), np.array(hand_confidences))
    
    # TODO: Integration with SAM2 for object tracking
    # For M0 validation, we might just use the hand-object distance if we have ground truth
    # or if we manually prompt SAM2 on the first frame.
    
    # 2. Compute Hand-Object distance (if object tracking was done)
    # Placeholder for object traj
    obj_centroids = np.zeros((len(hand_landmarks), 2)) # Dummy
    obj_traj = ObjectTrajectory(obj_centroids)
    
    # 3. Visualization
    output_video = os.path.join(output_dir, f"{basename}_processed.mp4")
    with VideoWriter(output_video, fps, width, height) as writer:
        reader.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i, frame in enumerate(reader.read_frames()):
            # Draw hand
            frame = draw_landmarks(frame, hand_traj.landmarks[i], hand_traj.confidences[i])
            # Draw object (placeholder)
            writer.write_frame(frame)
            
    print(f"Results saved to {output_dir}")
    mp_tracker.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--out", type=str, default="results", help="Output directory")
    args = parser.parse_args()
    
    run_extraction(args.video, args.out)
