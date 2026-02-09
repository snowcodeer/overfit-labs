import os
import boto3
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class BlobStorage:
    def __init__(self):
        self.bucket_name = os.getenv("BLOB_BUCKET_NAME")
        self.endpoint_url = os.getenv("BLOB_ENDPOINT_URL")
        self.access_key = os.getenv("BLOB_ACCESS_KEY")
        self.secret_key = os.getenv("BLOB_SECRET_KEY")
        
        if all([self.bucket_name, self.access_key, self.secret_key]):
            self.s3 = boto3.client(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key
            )
            self.enabled = True
            print(f"Cloud Storage initialized: {self.bucket_name}")
        else:
            self.s3 = None
            self.enabled = False
            print("Cloud Storage disabled: Missing credentials in .env")

    def upload_file(self, local_path: str, remote_path: str):
        if not self.enabled: return
        try:
            self.s3.upload_file(local_path, self.bucket_name, remote_path)
            # print(f"Uploaded {local_path} -> {remote_path}")
        except Exception as e:
            print(f"Upload failed: {e}")

    def upload_dir(self, local_dir: str, remote_prefix: str):
        if not self.enabled: return
        local_path = Path(local_dir)
        for path in local_path.rglob('*'):
            if path.is_file():
                # Construct remote path
                rel_path = path.relative_to(local_path)
                remote_path = f"{remote_prefix}/{rel_path.as_posix()}"
                self.upload_file(str(path), remote_path)

    def download_file(self, remote_path: str, local_path: str):
        if not self.enabled: return
        try:
            self.s3.download_file(self.bucket_name, remote_path, local_path)
            print(f"Downloaded {remote_path} -> {local_path}")
        except Exception as e:
            print(f"Download failed: {e}")
            raise e

    def get_url(self, remote_path: str, expiration: int = 3600):
        """Generate a presigned URL for accessing a file in S3"""
        if not self.enabled:
            return None
        try:
            url = self.s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': remote_path},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            print(f"Failed to generate URL: {e}")
            return None

    def list_runs(self):
        if not self.enabled: return []
        # Logic to list unique folders in 'runs/' prefix
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix="runs/", Delimiter='/')
            prefixes = response.get('CommonPrefixes', [])
            return [p['Prefix'] for p in prefixes]
        except Exception as e:
            print(f"List failed: {e}")
            return []

storage = BlobStorage()
