"""
Migrate existing data directory to task-based folder structure.

Old structure:
  data/
    video1.mp4
    video2.mp4
    analysis/
      video1_analysis.json
      video2_analysis.json
    labeled/
      video1_labeled.mp4
      video2_labeled.mp4

New structure:
  data/
    video1/
      video.mp4
      analysis.json
      labeled.mp4
    video2/
      video.mp4
      analysis.json
      labeled.mp4
"""

import shutil
from pathlib import Path

def migrate_data_structure():
    data_dir = Path("data")
    
    if not data_dir.exists():
        print("No data directory found")
        return
    
    # Find all video files in root data directory
    video_files = list(data_dir.glob("*.mp4")) + list(data_dir.glob("*.mov")) + list(data_dir.glob("*.avi"))
    
    print(f"Found {len(video_files)} videos to migrate")
    
    for video_path in video_files:
        task_name = video_path.stem
        task_dir = data_dir / task_name
        
        # Skip if already migrated
        if task_dir.exists() and (task_dir / "video.mp4").exists():
            print(f"Skipping {task_name} (already migrated)")
            continue
        
        print(f"\nMigrating: {task_name}")
        task_dir.mkdir(exist_ok=True)
        
        # Move video file
        new_video_path = task_dir / f"video{video_path.suffix}"
        if not new_video_path.exists():
            shutil.move(str(video_path), str(new_video_path))
            print(f"  ✓ Moved video to {new_video_path}")
        
        # Move analysis file if exists
        old_analysis = data_dir / "analysis" / f"{task_name}_analysis.json"
        new_analysis = task_dir / "analysis.json"
        if old_analysis.exists() and not new_analysis.exists():
            shutil.move(str(old_analysis), str(new_analysis))
            print(f"  ✓ Moved analysis to {new_analysis}")
        
        # Move labeled video if exists
        old_labeled = data_dir / "labeled" / f"{task_name}_labeled.mp4"
        new_labeled = task_dir / "labeled.mp4"
        if old_labeled.exists() and not new_labeled.exists():
            shutil.move(str(old_labeled), str(new_labeled))
            print(f"  ✓ Moved labeled video to {new_labeled}")
    
    # Clean up empty directories
    for old_dir in [data_dir / "analysis", data_dir / "labeled"]:
        if old_dir.exists() and not any(old_dir.iterdir()):
            old_dir.rmdir()
            print(f"\n✓ Removed empty directory: {old_dir}")
    
    print("\n✅ Migration complete!")

if __name__ == "__main__":
    migrate_data_structure()
