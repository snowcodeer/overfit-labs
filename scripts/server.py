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
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

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

REPLAYS_DIR = Path("data/replays")
ANALYSIS_DIR = Path("data/analysis")
REPLAYS_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# Folders to ignore when listing runs
RUN_BLACKLIST = {"plots", "tb", "tensorboard", "temp", "logs", "checkpoints", "runs"}

class RunInfo(BaseModel):
    id: str
    group: str
    timestamp: str
    config: dict
    has_history: bool
    has_video: bool

class ChatRequest(BaseModel):
    message: str
    current_analysis: dict

@app.get("/api/runs", response_model=List[RunInfo])
def list_runs():
    all_runs = []
    
    # 1. Local Runs
    if RUNS_DIR.exists():
        for group_dir in RUNS_DIR.iterdir():
            if group_dir.is_dir():
                for run_dir in group_dir.iterdir():
                    if run_dir.is_dir() and run_dir.name not in RUN_BLACKLIST and (run_dir / "config.pkl").exists():
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
                        
                        if run_id in RUN_BLACKLIST:
                            continue

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
    type: str = "train" # train, analyze, replay
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

    def add_job(self, video: str, resume_path: Optional[str] = None, timesteps: int = 50000, session_id: Optional[str] = None, job_type: str = "train") -> str:
        job_id = str(uuid.uuid4())[:8]
        job = Job(
            id=job_id,
            video=video,
            session_id=session_id,
            resume_path=resume_path,
            timesteps=timesteps,
            type=job_type,
            created_at=time.time()
        )
        with self.lock:
            self.jobs[job_id] = job
            self.queue.append(job_id)
        print(f"[Queue] Added {job_type} job {job_id} for {video}")
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
                    proc = getattr(job, "process", None)
                    if proc and proc.poll() is not None: # Process finished
                        return_code = proc.returncode
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
        
        if job.type == "analyze":
            cmd = [
                sys.executable, "scripts/analyze_and_label.py",
                "--video", job.video
            ]
        elif job.type == "replay":
            video_name = Path(job.video).stem
            analysis_path = ANALYSIS_DIR / f"{video_name}_analysis.json"
            output_path = REPLAYS_DIR / f"{video_name}_sim_sbs.mp4"
            cmd = [
                sys.executable, "scripts/replay_m1.py",
                "--video", job.video,
                "--analysis", str(analysis_path),
                "--out", str(output_path)
            ]
        else:
            cmd = [
                sys.executable, "scripts/train_grasp_residual.py",
                "--video", job.video,
                "--timesteps", str(job.timesteps)
            ]
            if job.resume_path:
                cmd.extend(["--resume", job.resume_path])

        print(f"[Queue] Starting {job.type} job {job_id}: {' '.join(cmd)}")
        try:
            # Use Popen to keep track of process
            process = subprocess.Popen(cmd)
            job.pid = process.pid
            # Store process object in job attribute 
            setattr(job, "process", process)
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
    type: str = "train"

@app.post("/api/queue/add")
def add_to_queue(req: EnqueueRequest):
    job_id = job_queue.add_job(req.video, req.resume_path, req.timesteps, req.session_id, req.type)
    return {"status": "queued", "job_id": job_id}

class ConfirmAnalysisRequest(BaseModel):
    video: str
    analysis: dict
    session_id: Optional[str] = None

@app.post("/api/analysis/confirm")
async def confirm_analysis(req: ConfirmAnalysisRequest):
    video_path = Path(req.video)
    analysis_file = ANALYSIS_DIR / f"{video_path.stem}_analysis.json"
    
    # Save confirmed analysis
    with open(analysis_file, "w") as f:
        json.dump(req.analysis, f, indent=4)
    
    # Trace: Enqueue Replay + Training
    # Replay job (MuJoCo sim)
    replay_id = job_queue.add_job(video=req.video, session_id=req.session_id, job_type="replay")
    
    # Training job
    train_id = job_queue.add_job(video=req.video, session_id=req.session_id, job_type="train")
    
    return {
        "status": "confirmed", 
        "replay_job_id": replay_id, 
        "train_job_id": train_id
    }

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


# --- Videos Library API ---
@app.get("/api/videos/library")
def get_videos_library():
    """Get list of all uploaded videos with their analysis status"""
    data_dir = Path("data")
    videos = []
    
    if not data_dir.exists():
        return {"videos": []}
    
    for task_dir in data_dir.iterdir():
        if not task_dir.is_dir() or task_dir.name.startswith('.'):
            continue
        
        video_file = task_dir / "video.mp4"
        analysis_file = task_dir / "analysis.json"
        
        if not video_file.exists():
            continue
        
        # Determine status
        status = "none"
        task_type = None
        if analysis_file.exists():
            status = "completed"
            try:
                with open(analysis_file) as f:
                    analysis_data = json.load(f)
                    task_type = analysis_data.get("task_type")
            except:
                pass
        
        # Generate S3 URL if storage is enabled, otherwise use local path
        video_url = str(video_file)
        if storage.enabled:
            # S3 URL format
            video_url = storage.get_url(f"data/{task_dir.name}/video.mp4")
        
        videos.append({
            "task_name": task_dir.name,
            "path": str(video_file),  # Keep local path for backend operations
            "video_url": video_url,   # URL for frontend to fetch from
            "status": status,
            "task_type": task_type,
            "created_at": video_file.stat().st_mtime
        })
    
    # Sort by creation time (newest first)
    videos.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {"videos": videos}

@app.delete("/api/videos/{task_name}")
def delete_video(task_name: str):
    """Delete a video and all associated files"""
    task_dir = Path("data") / task_name
    
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    try:
        shutil.rmtree(task_dir)
        return {"status": "deleted", "task_name": task_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {e}")

# --- Training API (for hybrid deployment) ---
@app.get("/api/training/prepare/{task_name}")
def prepare_training(task_name: str):
    """Get training configuration and download URLs for local training"""
    task_dir = Path("data") / task_name
    
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    
    analysis_file = task_dir / "analysis.json"
    if not analysis_file.exists():
        raise HTTPException(status_code=400, detail="Video not analyzed yet")
    
    # Load analysis
    with open(analysis_file) as f:
        analysis = json.load(f)
    
    # Generate S3 URLs if storage is enabled
    video_url = f"/data/{task_name}/video.mp4"
    analysis_url = f"/data/{task_name}/analysis.json"
    
    if storage.enabled:
        video_url = storage.get_url(f"data/{task_name}/video.mp4")
        analysis_url = storage.get_url(f"data/{task_name}/analysis.json")
    
    return {
        "task_name": task_name,
        "task_type": analysis.get("task_type", "unknown"),
        "video_url": video_url,
        "analysis_url": analysis_url,
        "training_command": f"python scripts/train_cli.py --task {task_name} --backend http://localhost:8000",
        "milestones": analysis.get("milestones", [])
    }

@app.post("/api/training/start/{task_name}")
def start_training(task_name: str, machine_info: dict = None):
    """Mark training as started"""
    # Update job queue status
    for job in job_queue:
        if job.get("session_id") == task_name:
            job["status"] = "training_local"
            job["machine_info"] = machine_info or {}
            break
    
    return {"status": "started", "task_name": task_name}

@app.post("/api/training/progress/{task_name}")
def update_training_progress(task_name: str, progress: dict):
    """Receive progress updates from local training"""
    # Update job queue with progress
    for job in job_queue:
        if job.get("session_id") == task_name:
            job["training_progress"] = progress
            break
    
    return {"status": "updated"}

@app.post("/api/training/complete/{task_name}")
def complete_training(task_name: str, results: dict):
    """Mark training as complete and validate results"""
    # Check if results were uploaded to S3
    run_dir = Path("runs") / task_name
    
    # Update job queue
    for job in job_queue:
        if job.get("session_id") == task_name:
            job["status"] = "completed"
            job["results"] = results
            break
    
    return {"status": "completed", "task_name": task_name}

@app.post("/api/training/chat/{task_name}")
async def chat_with_gemini(task_name: str, request: ChatRequest):
    """Refine labels via chat with Gemini"""
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key not configured")
    
    # Use flash for quick turnaround
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    prompt = f"""
    You are an expert robotics assistant helping to refine event labels (milestones) for a video-based training task.
    
    Task Name: {task_name}
    Current Analysis: {json.dumps(request.current_analysis, indent=2)}
    
    User Input: {request.message}
    
    Instructions:
    1. Based on the user input, adjust the milestone frames or labels in the analysis.
    2. If the user asks for a relative change (e.g. "shift everything by 5 frames"), calculate the new frame numbers.
    3. If the user asks for a new milestone, add it with a reasonable frame estimate if possible.
    4. Return ONLY a JSON object with two fields:
       - "analysis": the updated analysis JSON (matching the input task_type, milestones, etc. structure)
       - "thinking": a brief one-sentence explanation of what you changed.
    
    Example response structure:
    {{
        "analysis": {{ 
            "object_name": "...",
            "task_type": "...",
            "milestones": [...] 
        }},
        "thinking": "Updated the grasp_frame to 15 as requested."
    }}
    """
    
    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        return json.loads(response.text)
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# --- Upload Feature ---
from fastapi import UploadFile, File

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    # 1. Validate File Size (Max 50MB) 
    MAX_SIZE = 50 * 1024 * 1024
    
    # 2. Create task-based folder structure
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir()
    
    # Sanitize filename and create task folder
    safe_name = Path(file.filename).name.replace(" ", "_")
    task_name = Path(safe_name).stem  # filename without extension
    task_dir = data_dir / task_name
    task_dir.mkdir(exist_ok=True)
    
    # Save video as "video.mp4" in task folder
    video_ext = Path(safe_name).suffix
    target_path = task_dir / f"video{video_ext}"
    
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
    
    # Enqueue Analysis Job
    job_id = job_queue.add_job(video=str(target_path), job_type="analyze", session_id=task_name)
            
    return {"status": "uploaded", "filename": safe_name, "path": str(target_path), "task_name": task_name, "job_id": job_id}


@app.get("/api/train/status/{task_id}")
def get_train_status(task_id: str):
    return {"status": active_tasks.get(task_id, "not_found")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
