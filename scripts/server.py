import os
import pickle
import json
import uuid
import time
import threading
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from vidreward.utils.storage import storage
import subprocess
import sys
import shutil

app = FastAPI(title="OVERFIT Robotics Iteration Dashboard API")

# Enable CORS for the frontend (Vite defaults to 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RUNS_DIR = Path("runs")

# Create runs dir if not exists
if not RUNS_DIR.exists():
    RUNS_DIR.mkdir()

# Serve the runs directory as static files for video streaming
app.mount("/runs", StaticFiles(directory="runs"), name="runs")
app.mount("/data", StaticFiles(directory="data"), name="data")

class RunInfo(BaseModel):
    id: str
    group: str
    timestamp: str
    config: dict
    has_history: bool
    has_video: bool

@app.get("/api/runs", response_model=List[RunInfo])
def list_runs():
    all_runs = []
    
    # 1. Local Runs
    if RUNS_DIR.exists():
        for group_dir in RUNS_DIR.iterdir():
            if group_dir.is_dir():
                for run_dir in group_dir.iterdir():
                    if run_dir.is_dir() and (run_dir / "config.pkl").exists():
                        try:
                            # ... (existing loading logic) ...
                            with open(run_dir / "config.pkl", "rb") as f:
                                config = pickle.load(f)
                            
                            clean_config = {}
                            for k, v in config.items():
                                if hasattr(v, "tolist"): clean_config[k] = v.tolist()
                                else: clean_config[k] = v

                            all_runs.append(RunInfo(
                                id=run_dir.name,
                                group=group_dir.name,
                                timestamp=run_dir.name.split("_")[-1] if "_" in run_dir.name else "",
                                config=clean_config,
                                has_history=(run_dir / "plots" / "history.pkl").exists(),
                                has_video=len(list(run_dir.glob("*.mp4"))) > 0
                            ))
                        except: pass

    # 2. Cloud Runs (if enabled)
    if storage.enabled:
        try:
            cloud_prefixes = storage.list_runs()
            # cloud_prefixes are like ['runs/group/', 'runs/other_group/']
            # We need to iterate deeper to find runs if list_runs only returns groups
            # Actually, let's look at storage.py: it returns CommonPrefixes for 'runs/'
            for group_prefix in cloud_prefixes:
                group_name = group_prefix.strip("/").split("/")[-1]
                
                # List runs in this group
                try:
                    res = storage.s3.list_objects_v2(Bucket=storage.bucket_name, Prefix=group_prefix, Delimiter='/')
                    run_prefixes = res.get('CommonPrefixes', [])
                    for run_p in run_prefixes:
                        run_id = run_p['Prefix'].strip("/").split("/")[-1]
                        
                        # Avoid duplicates if local run exists
                        if any(r.id == run_id and r.group == group_name for r in all_runs):
                            continue
                            
                        # Add a placeholder RunInfo for cloud runs
                        all_runs.append(RunInfo(
                            id=run_id,
                            group=f"{group_name} (Cloud)",
                            timestamp=run_id.split("_")[-1] if "_" in run_id else "",
                            config={}, # We don't fetch full config from cloud for the list
                            has_history=True, # Assume true for cloud runs
                            has_video=False # Hard to tell without listing all files
                        ))
                except: pass
        except Exception as e:
            print(f"Cloud list failed: {e}")

    return all_runs

@app.get("/api/run/{group}/{run_id}/history")
def get_run_history(group: str, run_id: str):
    history_path = RUNS_DIR / group.replace(" (Cloud)", "") / run_id / "plots" / "history.pkl"
    if not history_path.exists():
        if "(Cloud)" in group:
            try:
                # Download from cloud
                clean_group = group.replace(" (Cloud)", "")
                remote_path = f"runs/{clean_group}/{run_id}/plots/history.pkl"
                os.makedirs(history_path.parent, exist_ok=True)
                print(f"Syncing {remote_path} from cloud...")
                storage.download_file(remote_path, str(history_path))
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Run is in cloud only and download failed: {e}")
        else:
            raise HTTPException(status_code=404, detail="History not found")
    
    with open(history_path, "rb") as f:
        history = pickle.load(f)
    
    return history

@app.get("/api/run/{group}/{run_id}/files")
def get_run_files(group: str, run_id: str):
    run_path = RUNS_DIR / group / run_id
    if not run_path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    
    files = {
        "videos": [f.name for f in run_path.glob("*.mp4")],
        "plots": [f.name for f in (run_path / "plots").glob("*.png")] if (run_path / "plots").exists() else []
    }
    return files

@app.get("/api/videos")
def list_videos():
    video_exts = [".mp4", ".mov", ".avi"]
    data_dir = Path("data")
    if not data_dir.exists():
        return []
    return [v.name for v in data_dir.iterdir() if v.suffix.lower() in video_exts]

class TrainRequest(BaseModel):
    video: str
    resume_path: Optional[str] = None
    timesteps: int = 50000

active_tasks: Dict[str, str] = {}

def run_training(video: str, resume_path: Optional[str], timesteps: int, task_id: str):
    import subprocess
    import sys
    
    cmd = [
        sys.executable, "scripts/train_grasp_residual.py",
        "--video", video,
        "--timesteps", str(timesteps)
    ]
    if resume_path:
        cmd.extend(["--resume", resume_path])
    
    print(f"Starting background training: {' '.join(cmd)}")
    active_tasks[task_id] = "running"
    try:
        subprocess.run(cmd, check=True)
        active_tasks[task_id] = "completed"
    except Exception as e:
        print(f"Training failed: {e}")
        active_tasks[task_id] = f"failed: {e}"

@app.post("/api/train/resume")
async def resume_training(req: TrainRequest, background_tasks: BackgroundTasks):
    job_id = job_queue.add_job(req.video, req.resume_path, req.timesteps)
    return {"status": "queued", "job_id": job_id}

# --- Job Queue System ---

class Job(BaseModel):
    id: str
    video: str
    session_id: Optional[str] = None # For filtering "my runs"
    resume_path: Optional[str] = None
    timesteps: int = 50000
    status: str = "pending" # pending, running, completed, failed, stopped
    created_at: float
    started_at: Optional[float] = None
    pid: Optional[int] = None

class JobQueue:
    def __init__(self, max_concurrent=3):
        self.jobs: Dict[str, Job] = {}
        self.queue: List[str] = [] # List of job IDs
        self.active: List[str] = [] # List of job IDs
        self.max_concurrent = max_concurrent
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def add_job(self, video: str, resume_path: Optional[str], timesteps: int, session_id: Optional[str] = None) -> str:
        job_id = str(uuid.uuid4())[:8]
        job = Job(
            id=job_id,
            video=video,
            session_id=session_id,
            resume_path=resume_path,
            timesteps=timesteps,
            created_at=time.time()
        )
        with self.lock:
            self.jobs[job_id] = job
            self.queue.append(job_id)
        print(f"[Queue] Added job {job_id} for {video}")
        return job_id

    def stop_job(self, job_id: str):
        with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                if job.status == "running" and job.pid:
                    print(f"[Queue] Stopping job {job_id} (PID {job.pid})...")
                    try:
                        # Windows specific kill
                        subprocess.run(["taskkill", "/F", "/T", "/PID", str(job.pid)])
                    except Exception as e:
                        print(f"Error killing job {job_id}: {e}")
                    job.status = "stopped"
                    if job_id in self.active:
                        self.active.remove(job_id)
                elif job.status == "pending":
                    job.status = "cancelled"
                    if job_id in self.queue:
                        self.queue.remove(job_id)

    def get_status(self):
        with self.lock:
            return {
                "queue": [self.jobs[jid] for jid in self.queue],
                "active": [self.jobs[jid] for jid in self.active],
                "history": [j for j in self.jobs.values() if j.status in ["completed", "failed", "stopped", "cancelled"]]
            }

    def _worker_loop(self):
        print("[Queue] Worker thread started.")
        while not self.stop_event.is_set():
            time.sleep(1)
            with self.lock:
                # Clean up finished jobs
                for job_id in list(self.active):
                    job = self.jobs[job_id]
                    if job.process.poll() is not None: # Process finished
                        return_code = job.process.return_code
                        if return_code == 0:
                            job.status = "completed"
                        else:
                            job.status = "failed"
                        self.active.remove(job_id)
                        print(f"[Queue] Job {job_id} finished with {job.status}")

                # Start new jobs if slots available
                while len(self.active) < self.max_concurrent and len(self.queue) > 0:
                    job_id = self.queue.pop(0)
                    self._start_job(job_id)

    def _start_job(self, job_id):
        job = self.jobs[job_id]
        job.status = "running"
        job.started_at = time.time()
        self.active.append(job_id)
        
        cmd = [
            sys.executable, "scripts/train_grasp_residual.py",
            "--video", job.video,
            "--timesteps", str(job.timesteps)
        ]
        if job.resume_path:
            cmd.extend(["--resume", job.resume_path])

        print(f"[Queue] Starting job {job_id}: {' '.join(cmd)}")
        try:
            # Use Popen to keep track of process
            process = subprocess.Popen(cmd)
            job.pid = process.pid
            # Store process object in job (not serializable, so transient)
            job.process = process 
        except Exception as e:
            print(f"[Queue] Failed to start job {job_id}: {e}")
            job.status = "failed"
            self.active.remove(job_id)


job_queue = JobQueue(max_concurrent=3)

class EnqueueRequest(BaseModel):
    video: str
    resume_path: Optional[str] = None
    timesteps: int = 50000
    session_id: Optional[str] = None

@app.post("/api/queue/add")
def add_to_queue(req: EnqueueRequest):
    job_id = job_queue.add_job(req.video, req.resume_path, req.timesteps, req.session_id)
    return {"status": "queued", "job_id": job_id}

@app.get("/api/queue/status")
def get_queue_status(session_id: Optional[str] = None):
    # Retrieve full status
    status = job_queue.get_status()
    
    # Optional filtering: 
    # If session_id is provided, you might want to mark which jobs belong to this session
    # For now, we return everything and let frontend filter, 
    # or we can add a flag "is_mine": True/False if we pass session_id here.
    # User asked for "tabs", so frontend filtering is easier effectively.
    return status

@app.post("/api/queue/stop/{job_id}")
def stop_queue_job(job_id: str):
    job_queue.stop_job(job_id)
    return {"status": "request_sent"}

# --- Download Feature ---

@app.get("/api/run/{group}/{run_id}/download")
def download_run(group: str, run_id: str):
    run_path = RUNS_DIR / group / run_id
    if not run_path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    
    zip_name = f"{run_id}.zip"
    zip_path = RUNS_DIR / "temp" / zip_name
    
    # Ensure temp dir exists
    (RUNS_DIR / "temp").mkdir(exist_ok=True)
    
    try:
        shutil.make_archive(str(zip_path.with_suffix('')), 'zip', run_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to zip run: {e}")

    from fastapi.responses import FileResponse
    return FileResponse(zip_path, filename=zip_name, media_type="application/zip")


# --- Upload Feature ---
from fastapi import UploadFile, File

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    # 1. Validate File Size (Max 50MB) 
    # Note: determining size streamingly is tricky, checking content-length header
    # or reading chunks.
    MAX_SIZE = 50 * 1024 * 1024
    
    # 2. Save file
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir()
    
    # Sanitize filename
    safe_name = Path(file.filename).name.replace(" ", "_")
    target_path = data_dir / safe_name
    
    size = 0
    with open(target_path, "wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024) # 1MB chunks
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_SIZE:
                target_path.unlink() # Delete partial
                raise HTTPException(status_code=413, detail="File too large (Max 50MB)")
            buffer.write(chunk)
            
    return {"status": "uploaded", "filename": safe_name, "path": f"data/{safe_name}"}


@app.get("/api/train/status/{task_id}")
def get_train_status(task_id: str):
    return {"status": active_tasks.get(task_id, "not_found")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
