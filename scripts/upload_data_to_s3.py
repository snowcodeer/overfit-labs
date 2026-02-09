"""
Upload all task directories to S3 bucket.
"""

from pathlib import Path
from vidreward.utils.storage import BlobStorage

def upload_all_data():
    storage = BlobStorage()
    
    if not storage.enabled:
        print("❌ S3 storage not configured. Check your .env file for:")
        print("  - BLOB_BUCKET_NAME")
        print("  - BLOB_ACCESS_KEY")
        print("  - BLOB_SECRET_KEY")
        print("  - BLOB_ENDPOINT_URL (optional)")
        return
    
    data_dir = Path("data")
    
    if not data_dir.exists():
        print("❌ No data directory found")
        return
    
    # Find all task directories
    task_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"Found {len(task_dirs)} task directories to upload")
    print(f"Uploading to bucket: {storage.bucket_name}\n")
    
    for task_dir in task_dirs:
        print(f"📤 Uploading: {task_dir.name}")
        try:
            storage.upload_dir(str(task_dir), f"data/{task_dir.name}")
            print(f"   ✅ Uploaded successfully\n")
        except Exception as e:
            print(f"   ❌ Failed: {e}\n")
    
    print("✅ Upload complete!")

if __name__ == "__main__":
    upload_all_data()
