"""
Analyze and Label Orchestrator
1. Runs Gemini Video Analyzer to get phases.
2. Runs Label Video to generate visualization.
"""

import sys
import subprocess
import argparse
from pathlib import Path
import json

def run_pipeline(video_path, output_dir=None):
    video_path = Path(video_path)
    
    # Determine task folder - if video is already in data/task_name/video.mp4, use that
    # Otherwise create a new task folder based on the video stem
    if video_path.parent.parent.name == "data":
        # Already in task folder structure: data/task_name/video.mp4
        task_dir = video_path.parent
    else:
        # Create task folder structure
        if not output_dir:
            output_dir = Path("data")
        else:
            output_dir = Path(output_dir)
        
        task_name = video_path.stem
        task_dir = output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
    
    analysis_json = task_dir / "analysis.json"
    labeled_video = task_dir / "labeled.mp4"
    
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    
    # 1. Analyze
    print(f"--- Step 1: Analyzing {video_path.name} ---")
    analyze_cmd = [
        sys.executable, "scripts/gemini_video_analyzer.py",
        "--video", str(video_path),
        "--output", str(analysis_json)
    ]
    subprocess.run(analyze_cmd, check=True, env=env)
    
    # 2. Label
    print(f"--- Step 2: Labeling {video_path.name} ---")
    label_cmd = [
        sys.executable, "scripts/label_video.py",
        "--video", str(video_path),
        "--analysis", str(analysis_json),
        "--output", str(labeled_video)
    ]
    subprocess.run(label_cmd, check=True, env=env)
    
    print(f"--- Pipeline Complete! ---")
    print(f"Task Directory: {task_dir}")
    print(f"Analysis: {analysis_json}")
    print(f"Labeled Video: {labeled_video}")
    
    # Upload to S3
    try:
        from vidreward.utils.storage import BlobStorage
        storage = BlobStorage()
        if storage.enabled:
            print(f"--- Uploading to S3 ---")
            # Upload entire task directory to S3
            storage.upload_dir(str(task_dir), f"data/{task_dir.name}")
            print(f"Uploaded task directory to S3: data/{task_dir.name}")
    except Exception as e:
        print(f"S3 upload failed (continuing anyway): {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output_dir", help="Directory for artifacts")
    
    args = parser.parse_args()
    try:
        run_pipeline(args.video, args.output_dir)
    except Exception as e:
        print(f"Error in pipeline: {e}")
        sys.exit(1)
