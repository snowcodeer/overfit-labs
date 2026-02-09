"""
Label Video Utility
Processes a video to overlay MediaPipe landmarks and Gemini-detected phases.
Outputs a side-by-side comparison video.
"""

import cv2
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from vidreward.extraction.mediapipe_tracker import MediaPipeTracker
from vidreward.utils.visualization import draw_landmarks, draw_phase_label

def label_video(video_path, analysis_path=None, output_path=None):
    video_path = Path(video_path)
    if not output_path:
        output_path = video_path.parent / f"{video_path.stem}_labeled.mp4"
    
    # Load analysis if available
    analysis = {}
    if analysis_path and Path(analysis_path).exists():
        with open(analysis_path, 'r') as f:
            analysis = json.load(f)
    
    milestones = analysis.get("milestones", [])
    
    # Fallback for old schema
    if not milestones:
        if "grasp_frame" in analysis:
            milestones.append({"label": "grasp", "frame": analysis["grasp_frame"]})
        if "release_frame" in analysis:
            milestones.append({"label": "release", "frame": analysis["release_frame"]})
    
    tracker = MediaPipeTracker()
    cap = cv2.VideoCapture(str(video_path))
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Output is single view (labeled)
    # Output is single view (labeled)
    out_size = (w, h)
    
    # Use mp4v as it is more reliable for OpenCV writers, though avc1 is better for browsers.
    # If the user has h264 perms, avc1 works. If not, mp4v works usually.
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, out_size)
        if not out.isOpened():
             print("Warning: mp4v failed, trying avc1...")
             fourcc = cv2.VideoWriter_fourcc(*'avc1')
             out = cv2.VideoWriter(str(output_path), fourcc, fps, out_size)
    except Exception as e:
        print(f"VideoWriter Init Failed: {e}")
    
    print(f"Labeling video: {video_path}")
    print(f"Output: {output_path}")
    
    print(f"Output: {output_path}")
    
    all_landmarks = []
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Labeled frame
        labeled_frame = frame.copy()
        
        # Track 
        lms, conf = tracker.process_frame(labeled_frame)
        if lms is not None:
            draw_landmarks(labeled_frame, lms, conf)
            # Store frame index and landmarks
            all_landmarks.append({
                "frame": frame_idx,
                "landmarks": lms,
                "confidence": conf
            })
            
        # Draw phases
        draw_phase_label(labeled_frame, frame_idx, milestones)
        
        # Write
        out.write(labeled_frame)
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx} frames...", end='\r')
            
    print(f"\nFinished! Processed {frame_idx} frames.")
    cap.release()
    out.release()
    tracker.close()

    # Save landmarks
    landmarks_path = video_path.parent / "landmarks.pkl"
    with open(landmarks_path, 'wb') as f:
        pickle.dump(all_landmarks, f)
    print(f"Saved landmarks to {landmarks_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--analysis", help="Path to video_analysis.json")
    parser.add_argument("--output", help="Output video path")
    
    args = parser.parse_args()
    label_video(args.video, args.analysis, args.output)
