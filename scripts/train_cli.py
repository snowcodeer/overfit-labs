"""
Training CLI Tool for Local GPU Training
Connects to cloud backend, downloads data, trains locally, uploads results.
"""

import argparse
import requests
import json
import subprocess
import sys
from pathlib import Path
import os

def download_file(url, output_path):
    """Download file from URL"""
    print(f"📥 Downloading {output_path.name}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✓ Downloaded {output_path.name}")

def upload_to_s3(local_dir, task_name):
    """Upload training results to S3"""
    try:
        from vidreward.utils.storage import storage
        if storage.enabled:
            print(f"📤 Uploading results to S3...")
            storage.upload_dir(str(local_dir), f"runs/{task_name}")
            print(f"✓ Results uploaded to S3")
            return True
    except Exception as e:
        print(f"⚠ Failed to upload to S3: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Train RL model locally with cloud backend")
    parser.add_argument("--task", required=True, help="Task name to train")
    parser.add_argument("--backend", default="http://localhost:8000", help="Backend URL")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--reward-config", help="Path to reward config JSON")
    
    args = parser.parse_args()
    
    backend_url = args.backend.rstrip('/')
    task_name = args.task
    
    print(f"\n🚀 OVERFIT Local Training")
    print(f"Task: {task_name}")
    print(f"Backend: {backend_url}\n")
    
    # Step 1: Get training configuration from backend
    print("1️⃣ Fetching training configuration...")
    try:
        response = requests.get(f"{backend_url}/api/training/prepare/{task_name}")
        response.raise_for_status()
        config = response.json()
    except Exception as e:
        print(f"❌ Failed to fetch configuration: {e}")
        sys.exit(1)
    
    print(f"✓ Task type: {config['task_type']}")
    print(f"✓ Milestones: {len(config['milestones'])}")
    
    # Step 2: Download data
    print("\n2️⃣ Downloading training data...")
    data_dir = Path("data") / task_name
    data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        download_file(config['video_url'], data_dir / "video.mp4")
        download_file(config['analysis_url'], data_dir / "analysis.json")
    except Exception as e:
        print(f"❌ Failed to download data: {e}")
        sys.exit(1)
    
    # Step 3: Notify backend that training is starting
    print("\n3️⃣ Starting training...")
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "CPU"
        
        machine_info = {
            "gpu": gpu_name,
            "cuda_available": gpu_available
        }
        
        requests.post(
            f"{backend_url}/api/training/start/{task_name}",
            json=machine_info
        )
        
        print(f"✓ GPU: {gpu_name}")
    except Exception as e:
        print(f"⚠ Could not detect GPU: {e}")
    
    # Step 4: Run training
    print("\n4️⃣ Running training on local GPU...")
    print("=" * 60)
    
    training_cmd = [
        sys.executable,
        "scripts/train_grasp_residual.py",
        "--session-id", task_name,
        "--epochs", str(args.epochs),
        "--video-path", str(data_dir / "video.mp4"),
        "--analysis-path", str(data_dir / "analysis.json")
    ]
    
    if args.reward_config:
        training_cmd.extend(["--reward-config", args.reward_config])
    
    try:
        result = subprocess.run(training_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        sys.exit(1)
    
    print("=" * 60)
    print("✓ Training completed!")
    
    # Step 5: Upload results to S3
    print("\n5️⃣ Uploading results...")
    run_dir = Path("runs") / task_name
    
    if run_dir.exists():
        upload_to_s3(run_dir, task_name)
    
    # Step 6: Notify backend of completion
    print("\n6️⃣ Notifying backend...")
    try:
        results = {
            "epochs_completed": args.epochs,
            "run_dir": str(run_dir)
        }
        
        requests.post(
            f"{backend_url}/api/training/complete/{task_name}",
            json=results
        )
        print("✓ Backend notified")
    except Exception as e:
        print(f"⚠ Failed to notify backend: {e}")
    
    print(f"\n🎉 Training complete! View results at {backend_url}")

if __name__ == "__main__":
    main()
