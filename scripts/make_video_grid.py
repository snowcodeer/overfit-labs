"""
M6 Visualization: Create Video Grid

Stitches multiple videos side-by-side for comparison.
Usage:
    python scripts/make_video_grid.py --videos real.mp4 vidreward.mp4 baseline.mp4 --labels "Real" "VidReward" "Baseline" --output comparison.mp4
"""

import cv2
import argparse
import numpy as np
from pathlib import Path

def resize_to_height(img, target_height):
    h, w = img.shape[:2]
    aspect = w / h
    target_width = int(target_height * aspect)
    return cv2.resize(img, (target_width, target_height))

def add_label(img, text):
    h, w = img.shape[:2]
    # Add black bar at top
    bar_h = 40
    new_img = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    new_img[bar_h:] = img
    
    cv2.putText(new_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return new_img

def make_grid(args):
    caps = [cv2.VideoCapture(v) for v in args.videos]
    
    # Check
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"Error opening {args.videos[i]}")
            return

    # Get properties
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    
    # Target height (min height of all videos)
    target_h = min([int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in caps])
    if target_h == 0: target_h = 480
    
    # Calculate output width
    total_w = 0
    for cap in caps:
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        aspect = w / h
        total_w += int(target_h * aspect)
        
    out_size = (total_w, target_h + 40) # +40 for label
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, out_size)
    
    print(f"Creating comparison video: {args.output}")
    print(f"Resolution: {out_size}")
    
    frame_count = 0
    while True:
        frames = []
        finished = False
        
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                finished = True
                break
            frames.append(frame)
            
        if finished:
            # Handle different lengths: loop or stop?
            # For comparison, usually stop when shortest ends
            break
            
        # Resize and Label
        processed = []
        for i, frame in enumerate(frames):
            resized = resize_to_height(frame, target_h)
            label = args.labels[i] if i < len(args.labels) else ""
            labeled = add_label(resized, label)
            processed.append(labeled)
            
        # Concatenate
        grid = np.concatenate(processed, axis=1)
        out.write(grid)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count/fps:.1f}s...", end='\r')
            
    print("\nDone!")
    for cap in caps: cap.release()
    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs='+', required=True)
    parser.add_argument("--labels", nargs='+', default=[])
    parser.add_argument("--output", default="comparison.mp4")
    
    args = parser.parse_args()
    make_grid(args)
