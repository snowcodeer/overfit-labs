import os
import argparse
import numpy as np
import cv2
from vidreward.extraction.mediapipe_tracker import MediaPipeTracker
from vidreward.extraction.trajectory import HandTrajectory, ObjectTrajectory
from vidreward.phases.phase_detector import PhaseDetector
from vidreward.phases.phase_features import compute_hand_object_distance
from vidreward.utils.video_io import VideoReader, VideoWriter
from vidreward.utils.visualization import draw_landmarks, plot_distances

def validate_m0(video_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(video_path).split('.')[0]
    
    # 1. Extraction
    print("Running extraction...")
    tracker = MediaPipeTracker()
    hand_traj = tracker.process_video(video_path)
    tracker.close()
    
    # 2. Mock Object Tracking (for M0 validation)
    # In a real scenario, SAM2 would provide this. 
    # Here we assume the object is at a fixed target or moving relative to hand
    num_frames = len(hand_traj.landmarks)
    # Mock: Object is at the end position of the hand (e.g., target location)
    obj_centroids = np.zeros((num_frames, 2))
    # Let's say the object is near the middle of the screen
    obj_centroids[:] = [0.5, 0.5] 
    obj_traj = ObjectTrajectory(obj_centroids)
    
    # 3. Phase Detection
    print("Detecting phases...")
    detector = PhaseDetector()
    phases = detector.detect_phases(hand_traj, obj_traj)
    
    # 4. Visualization
    print("Saving visualizations...")
    distances = compute_hand_object_distance(hand_traj, obj_traj)
    plot_distances(distances, os.path.join(output_dir, f"{basename}_distance.png"))
    
    # Save video with phase annotations
    reader = VideoReader(video_path)
    output_video = os.path.join(output_dir, f"{basename}_validated.mp4")
    
    with VideoWriter(output_video, reader.fps, reader.width, reader.height) as writer:
        for i, frame in enumerate(reader.read_frames()):
            # Draw landmarks
            frame = draw_landmarks(frame, hand_traj.landmarks[i], hand_traj.confidences[i])
            
            # Find current phase
            current_phase = "IDLE"
            for p in phases:
                if p.start_frame <= i <= p.end_frame:
                    current_phase = p.label
                    break
            
            # Draw phase label
            cv2.putText(frame, f"Phase: {current_phase}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # Draw distance
            cv2.putText(frame, f"Dist: {distances[i]:.3f}", (50, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            writer.write_frame(frame)
    
    print(f"Validation complete! Check {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--out", type=str, default="validation_results")
    args = parser.parse_args()
    
    # Run validation
    validate_m0(args.video, args.out)
