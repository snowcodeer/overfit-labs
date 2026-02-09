from vidreward.utils.storage import storage
import os
from pathlib import Path

def main():
    if not storage.enabled:
        print("❌ Cloud Storage is NOT enabled. Please check your .env file and ensure BLOB_BUCKET_NAME, BLOB_ACCESS_KEY, and BLOB_SECRET_KEY are set.")
        return

    runs_dir = Path("runs")
    if not runs_dir.exists():
        print("❌ 'runs' directory not found.")
        return

    print(f"🚀 Starting full cloud sync of '{runs_dir}' to bucket '{storage.bucket_name}'...")
    
    # We use upload_dir from the storage utility
    try:
        storage.upload_dir(str(runs_dir), "runs")
        print("✅ Cloud sync completed successfully.")
    except Exception as e:
        print(f"❌ Sync failed: {e}")

if __name__ == "__main__":
    main()
