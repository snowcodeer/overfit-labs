import sys
from pathlib import Path
# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

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
from vidreward.utils.storage import storage
import subprocess
import shutil
from google import genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
client = None
if API_KEY:
    client = genai.Client(api_key=API_KEY)

app = FastAPI(title="OVERFIT Robotics Iteration Dashboard API")

# Enable CORS for the frontend (Vite defaults to 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
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
        "plots": [f.name for f in (run_path / "plots").glob("*.png")] if (run_path / "plots").exists() else [],
        "code": [f.name for f in (run_path / "code").glob("*.py")] if (run_path / "code").exists() else [],
        "checkpoints": [f.name for f in run_path.glob("*.zip") if "checkpoint" in f.name.lower() or "td3" in f.name.lower()],
    }
    return files


@app.get("/api/run/{group}/{run_id}/code")
def get_run_code(group: str, run_id: str):
    """
    Get all code files from a training run for LLM context.
    Returns the full content of all Python files in the code/ directory.
    """
    run_path = RUNS_DIR / group.replace(" (Cloud)", "") / run_id
    code_dir = run_path / "code"

    # Try to download from cloud if not found locally
    if not code_dir.exists() and "(Cloud)" in group:
        try:
            clean_group = group.replace(" (Cloud)", "")
            # Download code directory from cloud
            code_dir.mkdir(parents=True, exist_ok=True)
            for code_file in ["config.py", "baseline.py", "residual.py", "reward.py", "train.py"]:
                remote_path = f"runs/{clean_group}/{run_id}/code/{code_file}"
                local_path = code_dir / code_file
                try:
                    storage.download_file(remote_path, str(local_path))
                except:
                    pass  # File might not exist
        except Exception as e:
            print(f"Failed to download code from cloud: {e}")

    if not code_dir.exists():
        return {"code_files": {}, "config_json": None}

    code_files = {}
    for code_file in code_dir.glob("*.py"):
        try:
            code_files[code_file.name] = code_file.read_text()
        except Exception as e:
            code_files[code_file.name] = f"# Error reading file: {e}"

    # Also return the config.json snapshot if it exists
    config_json = None
    config_path = run_path / "config.json"
    if config_path.exists():
        try:
            import json
            with open(config_path) as f:
                config_json = json.load(f)
        except:
            pass

    return {
        "code_files": code_files,
        "config_json": config_json
    }

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
        elif job.type == "eval":
            # job.video contains the run directory path for eval jobs
            cmd = [
                sys.executable, "scripts/eval_residual.py",
                "--run-dir", job.video,
                "--episodes", str(job.timesteps),
                "--save-metrics"
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
            # Set cwd to project root (parent of backend/)
            project_root = Path(__file__).parent.parent
            # Create log file for job output
            log_file = project_root / f"job_{job_id}.log"
            # Set PYTHONPATH so scripts can import vidreward
            env = os.environ.copy()
            env["PYTHONPATH"] = str(project_root)
            with open(log_file, "w") as f:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(project_root),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env
                )
            job.pid = process.pid
            # Store process object in job attribute
            setattr(job, "process", process)
            setattr(job, "log_file", str(log_file))
            print(f"[Queue] Job {job_id} started with PID {process.pid}, log: {log_file}")
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




# --- Eval Endpoints ---

class EvalRequest(BaseModel):
    episodes: int = 3

@app.post("/api/run/{group}/{run_id}/eval")
def run_evaluation(group: str, run_id: str, req: EvalRequest):
    """Queue an evaluation job for a training run."""
    run_path = RUNS_DIR / group / run_id
    if not run_path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    
    # Check if model exists
    if not (run_path / "td3_final.zip").exists():
        # Try checkpoints
        checkpoints = list((run_path / "checkpoints").glob("*.zip")) if (run_path / "checkpoints").exists() else []
        if not checkpoints:
            raise HTTPException(status_code=400, detail="No trained model found")
    
    job_id = job_queue.add_job(
        video=str(run_path),
        timesteps=req.episodes,  # Reuse for episodes count
        session_id=f"{group}/{run_id}",
        job_type="eval"
    )
    return {"status": "queued", "job_id": job_id}


@app.get("/api/run/{group}/{run_id}/eval/results")
def get_eval_results(group: str, run_id: str):
    """Get evaluation results for a run."""
    run_path = RUNS_DIR / group / run_id
    if not run_path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    
    results_path = run_path / "eval_results.json"
    eval_videos = list(run_path.glob("eval_*.mp4"))
    
    results = None
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
    
    return {
        "status": "completed" if results else "not_run",
        "results": results,
        "videos": [v.name for v in eval_videos]
    }


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
        
        # Generate S3 URL if storage is enabled, otherwise use None (frontend handles local URL)
        video_url = None
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

class AnalysisUpdate(BaseModel):
    analysis: dict

@app.post("/api/analysis/update/{task_name}")
def update_analysis(task_name: str, update: AnalysisUpdate):
    """Update analysis and regenerate labeled video"""
    task_dir = Path("data") / task_name
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail="Task not found")
        
    analysis_path = task_dir / "analysis.json"
    
    # 1. Save new analysis
    with open(analysis_path, "w") as f:
        json.dump(update.analysis, f, indent=4)
    
    # 2. Regenerate labeled video
    video_path = task_dir / "video.mp4"
    labeled_path = task_dir / "labeled.mp4"
    
    print(f"Regenerating video for {task_name}...")
    cmd = [
        sys.executable, "scripts/label_video.py",
        "--video", str(video_path),
        "--analysis", str(analysis_path),
        "--output", str(labeled_path)
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to regenerate video: {e}")
    
    # 3. Sync to S3 if enabled
    if storage.enabled:
        try:
            storage.upload_file(str(analysis_path), f"data/{task_name}/analysis.json")
            storage.upload_file(str(labeled_path), f"data/{task_name}/labeled.mp4")
        except Exception as e:
            print(f"S3 sync warning: {e}")

    return {"status": "updated", "task_name": task_name}

@app.post("/api/analyze/{task_name}")
def trigger_analysis(task_name: str):
    """Trigger analysis for an existing video task"""
    task_dir = Path("data") / task_name
    
    # Try common extensions
    video_path = None
    for ext in [".mp4", ".mov", ".avi"]:
        p = task_dir / f"video{ext}"
        if p.exists():
            video_path = p
            break
            
    if not video_path:
        raise HTTPException(status_code=404, detail="Video not found")
    
    job_id = job_queue.add_job(video=str(video_path), job_type="analyze", session_id=task_name)
    return {"status": "queued", "job_id": job_id, "task_name": task_name}


class ExperimentChatRequest(BaseModel):
    task_name: str
    message: str
    history: List[dict] = []
    run_context: Optional[dict] = None

@app.post("/api/experiment/chat")
async def experiment_chat_endpoint(req: ExperimentChatRequest):
    """
    Chat with context of the specific task (analysis + landmarks summary).
    """
    task_dir = Path("data") / req.task_name
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail="Task not found")

    # Load Context
    analysis = {}
    analysis_path = task_dir / "analysis.json"
    if analysis_path.exists():
        with open(analysis_path) as f:
            analysis = json.load(f)
            
    landmarks_summary = "No landmarks data found."
    landmarks_path = task_dir / "landmarks.pkl"
    if landmarks_path.exists():
        try:
            with open(landmarks_path, 'rb') as f:
                lms_data = pickle.load(f)
            if lms_data and len(lms_data) > 0:
                num_frames = len(lms_data)
                sample_keys = list(lms_data[0].keys())
                landmarks_summary = f"Landmarks available for {num_frames} frames. Keys per frame: {sample_keys}"
        except Exception as e:
            landmarks_summary = f"Error reading landmarks: {str(e)}"

    # Load the generated script template for context
    task_type = analysis.get("task_type", "pick")
    milestones = analysis.get("milestones", [])
    reward_config = analysis.get("reward_config", {})

    # Add run context if provided
    run_context_str = ""
    if req.run_context:
        run_id = req.run_context.get("run_id", "unknown")
        run_config = req.run_context.get("config", {})
        run_stats = req.run_context.get("stats")
        run_context_str = f"""
CURRENT EXPERIMENT RUN:
- Run ID: {run_id}
- Code Version: {run_config.get('code_version', 'N/A')}
- Reward Version: {run_config.get('reward_version', 'N/A')}"""
        if run_stats:
            run_context_str += f"""
- Total Episodes: {run_stats.get('total_episodes', 0)}
- Recent Rewards: {run_stats.get('recent_rewards', [])}
- Recent Successes: {run_stats.get('recent_successes', [])}"""

    # Construct System Prompt
    system_prompt = f"""You are an expert Robotics RL Experiment Designer helping users customize their training.

TASK CONTEXT:
- Task Name: {req.task_name}
- Task Type: {task_type}
- Milestones: {json.dumps(milestones, indent=2)}
- Current Reward Config: {json.dumps(reward_config, indent=2)}
- Mediapipe Data: {landmarks_summary}
{run_context_str}

MUJOCO ENVIRONMENT (AdroitHandRelocate-v1):
The robot is an Adroit Shadow Hand manipulating a cube object.

Key Bodies:
- "palm" - the palm of the hand
- "Object" - the cube being manipulated
- Fingertips: "ffdistal", "mfdistal", "rfdistal", "lfdistal", "thdistal" (ff=first, mf=middle, rf=ring, lf=little, th=thumb)

Key Geoms (for contact detection):
- Object: "cube"
- Palm: "C_palm0", "C_palm1"
- Finger collision geoms: "C_ffproximal", "C_ffmiddle", "C_ffdistal" (first finger), similarly C_mf*, C_rf*, C_lf* for other fingers
- Thumb: "C_thproximal", "C_thmiddle", "C_thdistal"

How to detect hand-object contact:
```python
def is_hand_contacting_object(data, model):
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        geoms = {{geom1_name, geom2_name}}
        if "cube" in geoms and any(g and g.startswith("C_") for g in geoms):
            return True
    return False
```

THE TRAINING SCRIPT:
A training script has been generated for this task using TD3 (Twin Delayed DDPG). The key section to edit is the `compute_reward()` function.

Available variables in compute_reward():
- `env.unwrapped.data`: MuJoCo simulation data (positions, velocities, contacts)
- `env.unwrapped.model`: MuJoCo model (body IDs, geom IDs, joint info)
- `env.target_pos`: Target position for the object (numpy array [x, y, z])
- `env.initial_obj_pos`: Starting position of the object
- `data.ncon`: Number of active contacts
- `data.contact[i]`: Contact info (geom1, geom2, pos, etc.)
- `data.xpos[model.body("palm").id]`: Palm position
- `data.xpos[model.body("Object").id]`: Object position
- `REWARD_CONFIG`: Dictionary of reward weights that can be tuned

Common reward terms:
- Time penalty: Small negative reward each step to encourage efficiency
- Contact reward: Reward for touching the object (use geom contact detection above)
- Lift bonus: Reward when object is lifted (obj_pos[2] > threshold)
- Transport reward: Reward for moving object toward target
- Success bonus: Large reward when task is completed
- Drop penalty: Negative reward if object is dropped after being lifted

YOUR ROLE:
1. Help users understand and modify the reward function
2. Suggest reward config values based on their goals
3. Provide Python code snippets for custom reward logic
4. Explain trade-offs of different reward designs
5. The algorithm is always TD3 unless the user specifies otherwise

IMPORTANT - CODE OUTPUT RULES:
- When providing code updates, ALWAYS output the COMPLETE file content, not just snippets
- For reward.py: include the full file with imports, docstring, and complete compute_reward() function
- For config.py: include the complete REWARD_CONFIG and MILESTONES dictionaries
- Wrap code in ```python``` blocks
- The user's editor will REPLACE the existing file with your output, so partial snippets will break things
- Focus on the compute_reward() function and REWARD_CONFIG dictionary
"""

    # Call Gemini
    try:
        if not client:
            raise ValueError("Gemini Client not initialized (Missing API Key)")

        chat = client.chats.create(
            model='models/gemini-3-flash-preview',
            history=[
                {"role": "user" if msg["role"] == "user" else "model", "parts": [{"text": msg["content"]}]}
                for msg in req.history
            ]
        )
        
        response = chat.send_message(system_prompt + "\n\nUser: " + req.message)
        return {"response": response.text}
        
    except Exception as e:
        print(f"Gemini Chat Error: {e}")
        # Fallback if Gemini fails or not configured
        return {"response": f"Error calling Gemini: {str(e)}. (Mock) I see you want to discuss {req.task_name}. The analysis shows {len(analysis.get('milestones', []))} milestones."}


class ForkExperimentRequest(BaseModel):
    base_run_group: str
    base_run_id: str
    task_name: str
    code_files: dict  # filename -> content
    launch_training: bool = False

@app.post("/api/experiment/fork")
async def fork_experiment(req: ForkExperimentRequest):
    """
    Fork an experiment with edited code files and optionally launch training.
    Creates a new run directory with the modified code.
    """
    from datetime import datetime
    import shutil

    # Validate base run exists
    base_run_dir = Path("runs") / req.base_run_group / req.base_run_id
    if not base_run_dir.exists():
        raise HTTPException(status_code=404, detail="Base run not found")

    # Create new run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_run_id = f"{req.task_name}_{timestamp}_fork"
    new_run_dir = Path("runs") / req.task_name / new_run_id
    new_run_dir.mkdir(parents=True, exist_ok=True)

    # Copy base config if exists
    base_config_path = base_run_dir / "config.pkl"
    if base_config_path.exists():
        shutil.copy(base_config_path, new_run_dir / "config.pkl")
        # Also update the config with new metadata
        try:
            with open(new_run_dir / "config.pkl", "rb") as f:
                config = pickle.load(f)
            config["code_version"] = f"v_{timestamp}_fork"
            config["forked_from"] = req.base_run_id
            config["task_name"] = req.task_name
            with open(new_run_dir / "config.pkl", "wb") as f:
                pickle.dump(config, f)
        except Exception as e:
            print(f"Warning: Could not update config: {e}")

    # Save edited code files
    code_dir = new_run_dir / "code"
    code_dir.mkdir(exist_ok=True)
    for filename, content in req.code_files.items():
        file_path = code_dir / filename
        with open(file_path, "w") as f:
            f.write(content)

    # Copy history if exists (for continuing training)
    base_history = base_run_dir / "plots" / "history.pkl"
    if base_history.exists():
        (new_run_dir / "plots").mkdir(exist_ok=True)
        shutil.copy(base_history, new_run_dir / "plots" / "history.pkl")

    result = {
        "status": "created",
        "new_run_id": new_run_id,
        "new_run_group": req.task_name,
        "run_dir": str(new_run_dir),
        "code_files_saved": list(req.code_files.keys())
    }

    # Optionally launch training
    if req.launch_training:
        video_path = f"data/{req.task_name}/video.mp4"
        if Path(video_path).exists():
            job_id = job_queue.add_job(
                video=video_path,
                job_type="train",
                resume_path=str(new_run_dir),
                session_id=req.task_name
            )
            result["job_id"] = job_id
            result["training_status"] = "queued"
        else:
            result["training_status"] = "skipped"
            result["training_error"] = f"Video not found: {video_path}"

    return result


@app.get("/api/experiment/notebook/{task_name}")
def generate_colab_notebook(task_name: str):
    """Generate a Colab-ready Jupyter notebook for the task."""
    from fastapi.responses import Response

    task_dir = Path("data") / task_name
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail="Task not found")

    analysis_path = task_dir / "analysis.json"
    if not analysis_path.exists():
        raise HTTPException(status_code=400, detail="Task not analyzed yet")

    with open(analysis_path) as f:
        analysis = json.load(f)

    task_type = analysis.get("task_type", "pick")
    milestones = analysis.get("milestones", [])
    reward_config = analysis.get("reward_config", {
        "success_bonus": 500.0,
        "lift_bonus": 5.0,
        "transport_scale": 10.0,
        "drop_penalty": -20.0,
        "time_penalty": -0.1,
        "contact_scale": 0.1,
    })

    # Build notebook cells
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"}
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# Training Script: {task_name}\n",
                    f"**Task Type:** {task_type}\n\n",
                    "This notebook trains a TD3 policy using residual RL for robotic manipulation.\n\n",
                    "## Setup\n",
                    "Run the cells below to install dependencies and configure training."
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Install dependencies\n",
                    "!pip install -q gymnasium gymnasium-robotics stable-baselines3 mujoco mediapy\n",
                    "!pip install -q google-generativeai  # For Gemini integration"
                ],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Mount Google Drive (for saving results)\n",
                    "from google.colab import drive\n",
                    "drive.mount('/content/drive')"
                ],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Download task data from your backend\n",
                    f"TASK_NAME = '{task_name}'\n",
                    f"BACKEND_URL = 'http://YOUR_BACKEND_URL:8000'  # Replace with your backend URL\n\n",
                    "import requests\n",
                    "from pathlib import Path\n\n",
                    "# Create data directory\n",
                    "data_dir = Path(f'data/{TASK_NAME}')\n",
                    "data_dir.mkdir(parents=True, exist_ok=True)\n\n",
                    "# Download video and analysis\n",
                    "# Note: Update these URLs to match your storage setup\n",
                    "print('Download your video.mp4 and analysis.json to:', data_dir)"
                ],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Task Configuration\n",
                    f"Detected milestones from video analysis:\n",
                    *[f"- **{m.get('label', 'unknown')}**: frame {m.get('frame', '?')}\n" for m in milestones]
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# ============================================================\n",
                    "# REWARD CONFIGURATION - Edit these values to tune training\n",
                    "# ============================================================\n",
                    f"TASK_TYPE = '{task_type}'\n\n",
                    "REWARD_CONFIG = {\n",
                    *[f"    '{k}': {v},\n" for k, v in reward_config.items()],
                    "}"
                ],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# ============================================================\n",
                    "# REWARD FUNCTION - Edit this to change training behavior\n",
                    "# ============================================================\n",
                    "import numpy as np\n\n",
                    "def compute_reward(env, obs, action, info):\n",
                    "    '''\n",
                    "    Compute reward for current step.\n",
                    "    \n",
                    "    Available:\n",
                    "      - env.unwrapped.data: MuJoCo state\n",
                    "      - env.unwrapped.model: MuJoCo model\n",
                    "      - env.target_pos: Target position [x,y,z]\n",
                    "      - REWARD_CONFIG: Weights dict\n",
                    "    '''\n",
                    "    data = env.unwrapped.data\n",
                    "    model = env.unwrapped.model\n",
                    "    \n",
                    "    obj_pos = data.xpos[model.body('Object').id].copy()\n",
                    "    dist_to_target = np.linalg.norm(obj_pos - env.target_pos)\n",
                    "    \n",
                    "    is_lifted = obj_pos[2] > 0.08\n",
                    "    n_contacts = data.ncon\n",
                    "    success = dist_to_target < 0.1\n",
                    "    \n",
                    "    # Build reward\n",
                    "    reward = REWARD_CONFIG['time_penalty']\n",
                    "    reward += min(n_contacts, 5) * REWARD_CONFIG['contact_scale']\n",
                    "    \n",
                    "    if is_lifted:\n",
                    "        reward += REWARD_CONFIG['lift_bonus']\n",
                    "        reward += (1.0 - np.clip(dist_to_target, 0, 1)) * REWARD_CONFIG['transport_scale']\n",
                    "    \n",
                    "    if success:\n",
                    "        reward += REWARD_CONFIG['success_bonus']\n",
                    "    \n",
                    "    dropped = getattr(env, 'ever_lifted', False) and not is_lifted\n",
                    "    if dropped:\n",
                    "        reward += REWARD_CONFIG['drop_penalty']\n",
                    "    \n",
                    "    return reward, {'success': success, 'lifted': is_lifted, 'dropped': dropped}"
                ],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# ============================================================\n",
                    "# TRAINING LOOP (TD3)\n",
                    "# ============================================================\n",
                    "import gymnasium as gym\n",
                    "import gymnasium_robotics\n",
                    "from stable_baselines3 import TD3\n",
                    "from stable_baselines3.common.noise import NormalActionNoise\n",
                    "from stable_baselines3.common.monitor import Monitor\n",
                    "from datetime import datetime\n\n",
                    "# Create environment\n",
                    "env = gym.make('AdroitHandRelocate-v1')\n",
                    "env = Monitor(env)\n\n",
                    "# TD3 Setup\n",
                    "n_actions = env.action_space.shape[-1]\n",
                    "action_noise = NormalActionNoise(\n",
                    "    mean=np.zeros(n_actions),\n",
                    "    sigma=0.1 * np.ones(n_actions)\n",
                    ")\n\n",
                    "model = TD3(\n",
                    "    'MlpPolicy',\n",
                    "    env,\n",
                    "    learning_rate=3e-4,\n",
                    "    buffer_size=100_000,\n",
                    "    learning_starts=1000,\n",
                    "    batch_size=256,\n",
                    "    action_noise=action_noise,\n",
                    "    policy_kwargs=dict(net_arch=[256, 256]),\n",
                    "    verbose=1\n",
                    ")\n\n",
                    "print('Starting training...')\n",
                    "model.learn(total_timesteps=50000, progress_bar=True)\n",
                    "print('Training complete!')"
                ],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Save model\n",
                    f"model.save('/content/drive/MyDrive/overfit_runs/{task_name}_td3')\n",
                    "print('Model saved to Google Drive!')"
                ],
                "execution_count": None,
                "outputs": []
            }
        ]
    }

    notebook_json = json.dumps(notebook, indent=2)

    return Response(
        content=notebook_json,
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=train_{task_name}.ipynb"
        }
    )


@app.get("/api/experiment/package/{task_name}")
def generate_training_package(task_name: str):
    """Generate a zip file with all components for local training."""
    import zipfile
    import io
    from fastapi.responses import StreamingResponse

    task_dir = Path("data") / task_name
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail="Task not found")

    analysis_path = task_dir / "analysis.json"
    if not analysis_path.exists():
        raise HTTPException(status_code=400, detail="Task not analyzed yet")

    # Get generated script
    script_response = generate_training_script(task_name)
    script_content = script_response["content"]

    # Create zip in memory
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add the training script
        zf.writestr(f"train_{task_name}.py", script_content)

        # Add analysis.json
        with open(analysis_path) as f:
            zf.writestr(f"data/{task_name}/analysis.json", f.read())

        # Add video if exists
        video_path = task_dir / "video.mp4"
        if video_path.exists():
            zf.write(video_path, f"data/{task_name}/video.mp4")

        # Add landmarks if exists
        landmarks_path = task_dir / "landmarks.pkl"
        if landmarks_path.exists():
            zf.write(landmarks_path, f"data/{task_name}/landmarks.pkl")

        # Add requirements.txt
        requirements = """gymnasium>=0.29.0
gymnasium-robotics>=1.2.0
stable-baselines3>=2.0.0
mujoco>=3.0.0
numpy>=1.24.0
torch>=2.0.0
mediapipe>=0.10.0
opencv-python>=4.8.0
"""
        zf.writestr("requirements.txt", requirements)

        # Add README
        readme = f"""# Training Package: {task_name}

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run training:
   ```bash
   python train_{task_name}.py
   ```

## Customizing the Reward

Edit the `compute_reward()` function in `train_{task_name}.py` to change training behavior.

Key variables available:
- `REWARD_CONFIG`: Dictionary of reward weights
- `env.target_pos`: Target position for the object
- `env.unwrapped.data`: MuJoCo simulation state

## Files Included

- `train_{task_name}.py` - Main training script (edit reward here)
- `data/{task_name}/video.mp4` - Source demonstration video
- `data/{task_name}/analysis.json` - Video analysis with milestones
- `requirements.txt` - Python dependencies
"""
        zf.writestr("README.md", readme)

    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename={task_name}_training.zip"
        }
    )


@app.get("/api/experiment/script/train")
def get_training_script():
    """Return the original training script content"""
    script_path = Path("scripts/train_grasp_residual.py")
    if not script_path.exists():
        raise HTTPException(status_code=404, detail="Training script not found")

    with open(script_path, "r") as f:
        content = f.read()

    return {"content": content}


@app.post("/api/experiment/generate-reward/{task_name}")
async def generate_reward_function(task_name: str):
    """
    Use Gemini to generate a custom reward function based on task analysis.
    """
    task_dir = Path("data") / task_name
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail="Task not found")

    analysis_path = task_dir / "analysis.json"
    if not analysis_path.exists():
        raise HTTPException(status_code=400, detail="Task not analyzed yet")

    with open(analysis_path) as f:
        analysis = json.load(f)

    task_type = analysis.get("task_type", "unknown")
    milestones = analysis.get("milestones", [])

    milestones_desc = "\n".join([
        f"- {m.get('label')}: frame {m.get('frame')} - {m.get('description', '')}"
        for m in milestones
    ])

    prompt = f"""Generate a Python reward function for a robotic manipulation task.

TASK: {task_name}
TYPE: {task_type}

MILESTONES DETECTED FROM VIDEO:
{milestones_desc}

AVAILABLE VARIABLES in compute_reward():
- data = env.unwrapped.data  # MuJoCo simulation data
- model = env.unwrapped.model  # MuJoCo model
- obj_pos = data.xpos[model.body("Object").id]  # Object position [x,y,z]
- palm_pos = data.xpos[model.body("palm").id]  # Palm position [x,y,z]
- obj_height = obj_pos[2]  # Object height
- obj_velocity = np.linalg.norm(data.qvel[-6:-3])  # Object velocity magnitude
- n_contacts = data.ncon  # Number of contact points
- dist_to_target = np.linalg.norm(obj_pos - env.target_pos)  # Distance to goal
- hand_to_obj = np.linalg.norm(palm_pos - obj_pos)  # Hand to object distance
- REWARD_CONFIG  # Dict with reward weights

REQUIREMENTS:
1. Generate ONLY the compute_reward function body (no imports, no function signature)
2. Use REWARD_CONFIG for all reward weights (e.g., REWARD_CONFIG["hold_bonus"])
3. Track milestone completion using: completed = getattr(env, "_completed_milestones", set())
4. For each milestone, check a physical condition and give bonus when achieved
5. Return (reward, info_dict) where info_dict has success, completed_milestones, etc.
6. Be creative with the conditions based on the milestone descriptions!

Generate clean, well-commented Python code:"""

    try:
        if not client:
            raise ValueError("Gemini client not initialized")

        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )

        reward_code = response.text

        # Clean up the response - extract just the code
        if "```python" in reward_code:
            reward_code = reward_code.split("```python")[1].split("```")[0]
        elif "```" in reward_code:
            reward_code = reward_code.split("```")[1].split("```")[0]

        return {"reward_code": reward_code.strip(), "milestones": milestones, "task_type": task_type}

    except Exception as e:
        print(f"Gemini reward generation failed: {e}")
        # Fallback to basic template
        return {
            "reward_code": generate_fallback_reward(milestones),
            "milestones": milestones,
            "task_type": task_type,
            "fallback": True
        }


def generate_fallback_reward(milestones):
    """Fallback reward if Gemini fails"""
    milestone_checks = []
    for m in milestones:
        label = m.get("label", "unknown")
        milestone_checks.append(f'    # {label} milestone\n    if "{label}" not in completed:\n        # TODO: Add condition for {label}\n        pass')

    return f'''# Base reward
reward = REWARD_CONFIG.get("time_penalty", -0.1)

# Track completed milestones
completed = getattr(env, "_completed_milestones", set())

# Contact reward
reward += min(n_contacts, 5) * REWARD_CONFIG.get("contact_scale", 0.1)

# Milestone rewards
{chr(10).join(milestone_checks)}

env._completed_milestones = completed

# Success bonus
success = dist_to_target < 0.1
if success:
    reward += REWARD_CONFIG.get("success_bonus", 500.0)

return reward, {{"success": success, "completed_milestones": list(completed)}}'''


class UploadTrainingRequest(BaseModel):
    task_name: str
    train_code: str
    config_code: str
    baseline_code: str = ""
    residual_code: str = ""
    reward_code: str


def get_next_reward_version(task_name: str) -> int:
    """Get the next reward version number by checking existing files in S3"""
    if not storage.enabled:
        return 1

    try:
        prefix = f"data/{task_name}/code/"
        response = storage.s3.list_objects_v2(Bucket=storage.bucket_name, Prefix=prefix)

        max_version = 0
        for obj in response.get('Contents', []):
            key = obj['Key']
            # Look for reward_v{n}.py pattern
            if 'reward_v' in key and key.endswith('.py'):
                try:
                    version = int(key.split('reward_v')[1].split('.py')[0])
                    max_version = max(max_version, version)
                except:
                    pass
        return max_version + 1
    except:
        return 1


@app.post("/api/experiment/upload-training")
async def upload_training_to_s3(req: UploadTrainingRequest):
    """
    Upload the full residual RL training files to S3 with versioning.
    Structure: data/{task_name}/code/
      - train_{timestamp}.py - Full pipeline
      - config_{timestamp}.py - All hyperparameters
      - baseline_{timestamp}.py - BC policy
      - residual_{timestamp}.py - Residual wrapper
      - reward_v1.py, reward_v2.py, etc. - Versioned rewards
    """
    if not storage.enabled:
        raise HTTPException(status_code=400, detail="S3 storage not configured")

    task_name = req.task_name
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    reward_version = get_next_reward_version(task_name)

    s3_prefix = f"data/{task_name}/code"

    try:
        import tempfile

        def upload_to_s3(content: str, filename: str):
            """Upload a file to S3"""
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                temp_path = f.name
            storage.upload_file(temp_path, f"{s3_prefix}/{filename}")
            os.unlink(temp_path)
            return filename

        # Local code directory - files saved with CLEAN names for execution
        local_code_dir = Path("data") / task_name / "code"
        local_code_dir.mkdir(parents=True, exist_ok=True)

        # Save locally with clean names (for `python train.py` to work)
        (local_code_dir / "train.py").write_text(req.train_code)
        (local_code_dir / "config.py").write_text(req.config_code)
        (local_code_dir / "baseline.py").write_text(req.baseline_code)
        (local_code_dir / "residual.py").write_text(req.residual_code)
        (local_code_dir / "reward.py").write_text(req.reward_code)

        # Also save versioned copies locally for history
        (local_code_dir / f"reward_v{reward_version}.py").write_text(req.reward_code)

        # Upload to S3: both clean names (latest) and versioned (history)
        # Clean names - always the "latest" version
        upload_to_s3(req.train_code, "train.py")
        upload_to_s3(req.config_code, "config.py")
        upload_to_s3(req.baseline_code, "baseline.py")
        upload_to_s3(req.residual_code, "residual.py")
        upload_to_s3(req.reward_code, "reward.py")

        # Versioned - for history tracking
        upload_to_s3(req.train_code, f"train_{timestamp}.py")
        upload_to_s3(req.config_code, f"config_{timestamp}.py")
        upload_to_s3(req.baseline_code, f"baseline_{timestamp}.py")
        upload_to_s3(req.residual_code, f"residual_{timestamp}.py")
        upload_to_s3(req.reward_code, f"reward_v{reward_version}.py")

        # Generate URLs for the clean-named files (what user will download/run)
        files = {
            "train": {"name": "train.py", "url": storage.get_url(f"{s3_prefix}/train.py")},
            "config": {"name": "config.py", "url": storage.get_url(f"{s3_prefix}/config.py")},
            "baseline": {"name": "baseline.py", "url": storage.get_url(f"{s3_prefix}/baseline.py")},
            "residual": {"name": "residual.py", "url": storage.get_url(f"{s3_prefix}/residual.py")},
            "reward": {"name": "reward.py", "url": storage.get_url(f"{s3_prefix}/reward.py")},
        }

        return {
            "status": "uploaded",
            "task_name": task_name,
            "timestamp": timestamp,
            "reward_version": reward_version,
            "files": files,
            "s3_prefix": s3_prefix,
            "local_path": str(local_code_dir)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


def generate_config_content(task_name: str, task_type: str, milestones: list, analysis: dict):
    """Generate the config.py content with ALL hyperparameters"""
    grasp_frame = analysis.get("grasp_frame", 0)
    release_frame = analysis.get("release_frame", 100)

    # Build milestone info for comments
    milestone_info = "\n".join([
        f"#   - {m.get('label', 'unknown')}: frame {m.get('frame', '?')} ({m.get('description', '')})"
        for m in milestones
    ])

    # Build dynamic reward config from milestones
    default_config = {
        "time_penalty": -0.1,
        "contact_scale": 0.1,
        "success_bonus": 500.0,
    }

    # Add bonus for each milestone
    for m in milestones:
        label = m.get("label", "unknown").lower().replace(" ", "_")
        default_config[f"{label}_bonus"] = 10.0

    # Override with analysis reward_config if present
    if "reward_config" in analysis:
        default_config.update(analysis["reward_config"])

    config_str = json.dumps(default_config, indent=4)

    # Build milestone frames dict
    milestone_frames = {m.get("label", "unknown"): m.get("frame", 0) for m in milestones}
    milestone_frames_str = json.dumps(milestone_frames, indent=4)

    return f'''"""
Configuration for: {task_name}
Task Type: {task_type}
Auto-generated from video analysis.

This file contains ALL hyperparameters for the residual RL pipeline:
- Task configuration (from video analysis)
- Reward configuration (milestone bonuses)
- BC (Behavioral Cloning) configuration
- Residual RL configuration
- Training configuration
"""

# ============================================================================
# TASK CONFIGURATION (from video analysis)
# ============================================================================
TASK_NAME = "{task_name}"
TASK_TYPE = "{task_type}"
GRASP_FRAME = {grasp_frame}
RELEASE_FRAME = {release_frame}

# Milestones detected from video:
{milestone_info}

MILESTONES = {milestone_frames_str}


# ============================================================================
# REWARD CONFIGURATION - Edit these to tune reward shaping
# ============================================================================
REWARD_CONFIG = {config_str}


# ============================================================================
# BC (BEHAVIORAL CLONING) CONFIGURATION - Tune the base policy
# ============================================================================
BC_CONFIG = {{
    # Network architecture
    "hidden_dims": (128, 128),      # MLP hidden layer sizes

    # Training
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 256,
    "epochs": 1000,
    "early_stop_patience": 50,

    # Data augmentation (critical for single-demo learning!)
    "augment_samples_per_frame": 50,  # Augmented samples per original frame
    "position_noise_std": 0.02,       # Position noise in meters (2cm)
    "joint_noise_std": 0.05,          # Joint angle noise in radians (~3 deg)
    "temporal_blend": True,           # Interpolate between consecutive frames
}}


# ============================================================================
# RESIDUAL RL CONFIGURATION - Tune the residual learning
# ============================================================================
RESIDUAL_CONFIG = {{
    # Residual scale curriculum (KEY hyperparameters!)
    # action = base_policy(s) + residual_scale * rl_policy(s)
    "initial_scale": 0.1,       # Start small - follow demo closely
    "final_scale": 0.5,         # End larger - allow bigger corrections
    "warmup_steps": 100_000,    # Steps to reach final scale

    # Residual penalty (keeps RL close to demo)
    "penalty_weight": 0.1,      # Weight for -||residual||^2 penalty
    "penalty_decay_rate": 0.99999,
    "min_penalty_weight": 0.01,

    # Action composition
    "use_delta_actions": True,  # True = delta actions, False = absolute
    "include_base_in_obs": True,  # Include base action in RL observation
}}


# ============================================================================
# TRAINING CONFIGURATION - Overall training settings
# ============================================================================
TRAINING_CONFIG = {{
    # Algorithm
    "algorithm": "TD3",
    "total_timesteps": 50_000,

    # TD3 hyperparameters
    "learning_rate": 3e-4,
    "buffer_size": 100_000,
    "learning_starts": 1000,
    "batch_size": 256,
    "tau": 0.005,               # Target network update rate
    "gamma": 0.99,              # Discount factor

    # Policy network
    "policy_arch": [256, 256],

    # Exploration noise
    "action_noise_sigma": 0.1,

    # Environment
    "max_episode_steps": 100,
}}
'''


def generate_baseline_content(task_name: str, task_type: str):
    """Generate the baseline.py content (BC policy for residual RL)"""
    return f'''"""
Behavioral Cloning Base Policy for: {task_name}
Task Type: {task_type}

This module provides the BC base policy that the residual RL learns on top of.
The key insight: BC from a single demo is fragile, but it gets us close.
RL learns small corrections to fix the errors.

Pipeline: Video -> MediaPipe -> Retarget -> BC Policy -> Residual RL
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple
from dataclasses import dataclass

from config import BC_CONFIG


@dataclass
class BCConfig:
    """Configuration for BC policy training."""
    hidden_dims: Tuple[int, ...] = BC_CONFIG.get("hidden_dims", (128, 128))
    learning_rate: float = BC_CONFIG.get("learning_rate", 1e-3)
    weight_decay: float = BC_CONFIG.get("weight_decay", 1e-4)
    batch_size: int = BC_CONFIG.get("batch_size", 256)
    epochs: int = BC_CONFIG.get("epochs", 1000)
    early_stop_patience: int = BC_CONFIG.get("early_stop_patience", 50)
    augment_samples_per_frame: int = BC_CONFIG.get("augment_samples_per_frame", 50)
    position_noise_std: float = BC_CONFIG.get("position_noise_std", 0.02)
    joint_noise_std: float = BC_CONFIG.get("joint_noise_std", 0.05)
    temporal_blend: bool = BC_CONFIG.get("temporal_blend", True)


def extract_features_from_sim(model, data) -> np.ndarray:
    """
    Extract relative features from MuJoCo state.

    Features:
    - hand_to_obj: (3,) relative position from palm to object
    - finger_joints: (24,) current finger joint angles

    Total: 27D state representation
    """
    # Get palm position
    try:
        palm_pos = data.xpos[model.body("palm").id].copy()
    except:
        palm_pos = data.qpos[:3].copy()

    # Get object position
    try:
        obj_pos = data.xpos[model.body("Object").id].copy()
    except:
        obj_pos = np.zeros(3)

    # Relative position (task-centric feature)
    hand_to_obj = obj_pos - palm_pos

    # Finger joints (indices 6-29 in qpos)
    finger_joints = data.qpos[6:30].copy()

    return np.concatenate([hand_to_obj, finger_joints])


class SingleDemoAugmenter:
    """
    Data augmentation for single-demonstration BC training.

    Key insight: When we add noise to the state, we keep the original action.
    This teaches the policy to "correct" back toward the demonstration.
    """

    def __init__(self, config: BCConfig = None):
        config = config or BCConfig()
        self.position_noise_std = config.position_noise_std
        self.joint_noise_std = config.joint_noise_std
        self.temporal_blend = config.temporal_blend

    def augment(self, states: np.ndarray, actions: np.ndarray,
                num_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Augment state-action pairs for single-demo learning."""
        T, state_dim = states.shape

        aug_states = [states.copy()]
        aug_actions = [actions.copy()]

        for _ in range(num_samples):
            noisy_states = states.copy()

            # Position noise (first 3 dims = hand_to_obj)
            noisy_states[:, :3] += np.random.normal(0, self.position_noise_std, (T, 3))

            # Joint noise (dims 3-27 = finger joints)
            noisy_states[:, 3:] += np.random.normal(0, self.joint_noise_std, (T, state_dim - 3))

            # Action stays the same - teaches correction!
            aug_states.append(noisy_states)
            aug_actions.append(actions.copy())

        # Temporal blending
        if self.temporal_blend and T > 1:
            for _ in range(num_samples // 2):
                alpha = np.random.uniform(0, 1, T - 1)
                blended_states = np.zeros((T - 1, state_dim))
                blended_actions = np.zeros((T - 1, actions.shape[1]))

                for t in range(T - 1):
                    a = alpha[t]
                    blended_states[t] = (1 - a) * states[t] + a * states[t + 1]
                    blended_actions[t] = (1 - a) * actions[t] + a * actions[t + 1]

                aug_states.append(blended_states)
                aug_actions.append(blended_actions)

        return np.vstack(aug_states), np.vstack(aug_actions)


class BCBasePolicy(nn.Module):
    """
    Behavioral Cloning policy network.

    Maps relative state features to delta joint actions.
    Intentionally small to prevent overfitting on single demo.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: Tuple[int, ...] = (128, 128)):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build MLP
        layers = []
        prev_dim = state_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
            ])
            prev_dim = h

        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())  # Bound outputs to [-1, 1]

        self.net = nn.Sequential(*layers)

        # Normalization stats
        self.register_buffer('state_mean', torch.zeros(state_dim))
        self.register_buffer('state_std', torch.ones(state_dim))
        self.register_buffer('action_scale', torch.ones(action_dim))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_normalization(self, state_mean, state_std, action_scale):
        self.state_mean = torch.FloatTensor(state_mean)
        self.state_std = torch.FloatTensor(state_std)
        self.action_scale = torch.FloatTensor(action_scale)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state_norm = (state - self.state_mean) / (self.state_std + 1e-8)
        action_norm = self.net(state_norm)
        return action_norm * self.action_scale

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Get action for a single state (numpy interface)."""
        self.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action_t = self.forward(state_t)
            return action_t.squeeze(0).numpy()


class BCTrainer:
    """Trainer for BC policy with augmentation and early stopping."""

    def __init__(self, config: BCConfig = None):
        self.config = config or BCConfig()
        self.augmenter = SingleDemoAugmenter(self.config)

    def collect_data_from_sim(self, env, joint_trajectory: np.ndarray):
        """
        Collect state-action pairs by replaying trajectory in simulation.

        Args:
            env: Adroit environment
            joint_trajectory: (T, 30) joint angles from retargeting

        Returns:
            states: (T-1, 27) relative features
            actions: (T-1, 30) delta joint angles
        """
        import mujoco

        model = env.unwrapped.model
        data = env.unwrapped.data

        T = len(joint_trajectory)
        states, actions = [], []

        env.reset()

        for t in range(T - 1):
            # Set sim to trajectory pose
            data.qpos[:30] = joint_trajectory[t]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)

            # Extract features
            state = extract_features_from_sim(model, data)
            states.append(state)

            # Delta action
            action = joint_trajectory[t + 1] - joint_trajectory[t]
            actions.append(action)

        return np.array(states), np.array(actions)

    def train(self, states: np.ndarray, actions: np.ndarray,
              verbose: bool = True) -> BCBasePolicy:
        """Train BC policy with augmentation and early stopping."""

        if verbose:
            print(f"Original data: {{len(states)}} samples")

        # Augment data
        aug_states, aug_actions = self.augmenter.augment(
            states, actions, self.config.augment_samples_per_frame
        )

        if verbose:
            print(f"After augmentation: {{len(aug_states)}} samples")

        # Train/val split
        n = len(aug_states)
        n_val = int(n * 0.1)
        indices = np.random.permutation(n)

        train_states = aug_states[indices[n_val:]]
        train_actions = aug_actions[indices[n_val:]]
        val_states = aug_states[indices[:n_val]]
        val_actions = aug_actions[indices[:n_val]]

        # Normalization
        state_mean = train_states.mean(axis=0)
        state_std = train_states.std(axis=0) + 1e-8
        action_scale = np.abs(train_actions).max(axis=0) + 1e-8

        # Create policy
        policy = BCBasePolicy(
            state_dim=states.shape[1],
            action_dim=actions.shape[1],
            hidden_dims=self.config.hidden_dims
        )
        policy.set_normalization(state_mean, state_std, action_scale)

        # Normalize data
        train_states_norm = (train_states - state_mean) / state_std
        train_actions_norm = train_actions / action_scale
        val_states_norm = (val_states - state_mean) / state_std
        val_actions_norm = val_actions / action_scale

        # DataLoader
        train_dataset = TensorDataset(
            torch.FloatTensor(train_states_norm),
            torch.FloatTensor(train_actions_norm)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

        val_states_t = torch.FloatTensor(val_states_norm)
        val_actions_t = torch.FloatTensor(val_actions_norm)

        # Optimizer
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_state_dict = None

        for epoch in range(self.config.epochs):
            policy.train()
            train_loss = 0.0

            for batch_states, batch_actions in train_loader:
                pred = policy.net(batch_states)
                loss = F.mse_loss(pred, batch_actions)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            policy.eval()
            with torch.no_grad():
                val_pred = policy.net(val_states_t)
                val_loss = F.mse_loss(val_pred, val_actions_t).item()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = {{k: v.clone() for k, v in policy.state_dict().items()}}
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {{epoch + 1}}: train={{train_loss:.6f}}, val={{val_loss:.6f}}")

            if patience_counter >= self.config.early_stop_patience:
                if verbose:
                    print(f"Early stopping at epoch {{epoch + 1}}")
                break

        if best_state_dict:
            policy.load_state_dict(best_state_dict)

        if verbose:
            print(f"BC training complete. Best val_loss: {{best_val_loss:.6f}}")

        return policy
'''


def generate_residual_content(task_name: str, task_type: str):
    """Generate the residual.py content (residual RL wrapper)"""
    return f'''"""
Residual RL Environment Wrapper for: {task_name}
Task Type: {task_type}

Implements the core residual RL paradigm:
    action = base_policy(s) + alpha * residual_policy(s)
             ↑ from BC           ↑ from TD3

The base policy (BC) gets us close to the demo.
The residual policy (RL) learns to correct errors.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional

from config import RESIDUAL_CONFIG
from baseline import BCBasePolicy, extract_features_from_sim


class ResidualRLEnv(gym.Wrapper):
    """
    Wraps environment to implement residual RL.

    action = base_action + residual_scale * residual_action

    Key features:
    - BC base policy provides demo-derived actions
    - Residual actions are bounded and scaled
    - Residual penalty keeps RL close to demo
    - Curriculum learning via residual_scale
    """

    def __init__(
        self,
        env: gym.Env,
        base_policy: BCBasePolicy,
        residual_scale: float = RESIDUAL_CONFIG.get("initial_scale", 0.1),
        residual_penalty_weight: float = RESIDUAL_CONFIG.get("penalty_weight", 0.1),
        use_delta_actions: bool = RESIDUAL_CONFIG.get("use_delta_actions", True),
        include_base_in_obs: bool = RESIDUAL_CONFIG.get("include_base_in_obs", True),
    ):
        super().__init__(env)

        self.base_policy = base_policy
        self.residual_scale = residual_scale
        self.residual_penalty_weight = residual_penalty_weight
        self.use_delta_actions = use_delta_actions
        self.include_base_in_obs = include_base_in_obs

        self._last_base_action = None
        self._last_residual_action = None

        # Modify obs space if including base action
        if include_base_in_obs:
            orig_space = env.observation_space
            base_dim = base_policy.action_dim

            low = np.concatenate([orig_space.low, np.full(base_dim, -np.inf)])
            high = np.concatenate([orig_space.high, np.full(base_dim, np.inf)])
            self.observation_space = spaces.Box(
                low=low.astype(np.float32),
                high=high.astype(np.float32)
            )

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)

        self._last_base_action = None
        self._last_residual_action = None

        if self.include_base_in_obs:
            features = self._get_features()
            base_action = self.base_policy.get_action(features)
            obs = np.concatenate([obs, base_action])
            self._last_base_action = base_action

        return obs, info

    def step(self, residual_action: np.ndarray):
        """
        Execute: base + scaled residual

        Args:
            residual_action: Output from RL policy

        Returns:
            Standard gym step outputs with modified reward
        """
        # 1. Get base action from BC
        features = self._get_features()
        base_action = self.base_policy.get_action(features)
        self._last_base_action = base_action

        # 2. Scale residual
        scaled_residual = np.clip(
            residual_action * self.residual_scale,
            -self.residual_scale,
            self.residual_scale
        )
        self._last_residual_action = scaled_residual

        # 3. Compose final action
        if self.use_delta_actions:
            current_qpos = self.env.unwrapped.data.qpos[:30].copy()
            target_qpos = current_qpos + base_action + scaled_residual
            final_action = np.clip(
                target_qpos,
                self.env.action_space.low,
                self.env.action_space.high
            )
        else:
            final_action = np.clip(
                base_action + scaled_residual,
                self.env.action_space.low,
                self.env.action_space.high
            )

        # 4. Step environment
        obs, env_reward, terminated, truncated, info = self.env.step(final_action)

        # 5. Residual penalty (keeps RL close to BC)
        residual_penalty = -self.residual_penalty_weight * np.sum(scaled_residual ** 2)

        # 6. Total reward
        total_reward = env_reward + residual_penalty

        # 7. Logging info
        info['base_action_norm'] = np.linalg.norm(base_action)
        info['residual_action_norm'] = np.linalg.norm(scaled_residual)
        info['residual_penalty'] = residual_penalty
        info['env_reward'] = env_reward
        info['residual_scale'] = self.residual_scale

        # 8. Augment obs
        if self.include_base_in_obs:
            next_features = self._get_features()
            next_base = self.base_policy.get_action(next_features)
            obs = np.concatenate([obs, next_base])

        return obs, total_reward, terminated, truncated, info

    def _get_features(self) -> np.ndarray:
        model = self.env.unwrapped.model
        data = self.env.unwrapped.data
        return extract_features_from_sim(model, data)

    def set_residual_scale(self, scale: float):
        """Update residual scale (for curriculum)."""
        self.residual_scale = scale

    def set_penalty_weight(self, weight: float):
        """Update residual penalty weight."""
        self.residual_penalty_weight = weight


class ResidualCurriculum:
    """
    Curriculum learning for residual scale.

    Early training: small scale -> follow demo closely
    Late training: larger scale -> allow corrections
    """

    def __init__(
        self,
        env: ResidualRLEnv,
        initial_scale: float = RESIDUAL_CONFIG.get("initial_scale", 0.1),
        final_scale: float = RESIDUAL_CONFIG.get("final_scale", 0.5),
        warmup_steps: int = RESIDUAL_CONFIG.get("warmup_steps", 100_000),
        penalty_decay: float = RESIDUAL_CONFIG.get("penalty_decay_rate", 0.99999),
        min_penalty: float = RESIDUAL_CONFIG.get("min_penalty_weight", 0.01),
    ):
        self.env = env
        self.initial_scale = initial_scale
        self.final_scale = final_scale
        self.warmup_steps = warmup_steps
        self.penalty_decay = penalty_decay
        self.min_penalty = min_penalty

        self.num_steps = 0
        self.initial_penalty = env.residual_penalty_weight

    def on_step(self) -> Dict[str, float]:
        """Called after each step. Returns metrics for logging."""
        self.num_steps += 1

        # Linear warmup for residual scale
        progress = min(1.0, self.num_steps / self.warmup_steps)
        new_scale = self.initial_scale + progress * (self.final_scale - self.initial_scale)
        self.env.set_residual_scale(new_scale)

        # Decay penalty
        new_penalty = max(
            self.min_penalty,
            self.initial_penalty * (self.penalty_decay ** self.num_steps)
        )
        self.env.set_penalty_weight(new_penalty)

        return {{
            'residual_scale': new_scale,
            'penalty_weight': new_penalty,
            'curriculum_progress': progress,
        }}
'''


def generate_reward_content(task_name: str, task_type: str, milestones: list):
    """Generate the reward.py content"""
    return f'''"""
Reward Function for: {task_name}
Task Type: {task_type}
Auto-generated from video analysis milestones.

Edit this file to customize the reward behavior!
"""

import numpy as np
from config import REWARD_CONFIG, MILESTONES


def compute_reward(env, obs, action, info):
    """
    Compute the reward for the current step.

    This is the main function to edit when tuning your training!

    Available variables:
        - env.unwrapped.data: MuJoCo simulation data
        - env.unwrapped.model: MuJoCo model
        - env.target_pos: Target position for the object (numpy array)
        - env.initial_obj_pos: Starting position of object
        - REWARD_CONFIG: Dictionary of reward weights (imported from config.py)
        - MILESTONES: Dictionary of milestone frames (imported from config.py)

    Args:
        env: The gym environment wrapper
        obs: Current observation (40D vector)
        action: Action taken (30D vector)
        info: Info dict from step

    Returns:
        tuple: (reward_float, info_dict)
    """
    data = env.unwrapped.data
    model = env.unwrapped.model

    # Get object and target positions
    obj_pos = data.xpos[model.body("Object").id].copy()
    dist_to_target = np.linalg.norm(obj_pos - env.target_pos)

    # Check conditions
    is_lifted = obj_pos[2] > 0.08  # Object above 8cm
    n_contacts = data.ncon
    success = dist_to_target < 0.1  # Within 10cm of target

    # --- BUILD REWARD ---
    reward = REWARD_CONFIG["time_penalty"]  # Small penalty each step

    # Contact reward (encourages touching the object)
    reward += min(n_contacts, 5) * REWARD_CONFIG["contact_scale"]

    # Get current phase based on simulation step
    current_step = getattr(env, "steps", 0)
    total_steps = getattr(env, "max_steps", 100)

    # Map sim step to video frame (approximate)
    total_frames = max(MILESTONES.values()) if MILESTONES else 100
    current_frame = int((current_step / total_steps) * total_frames)

    # Track milestone completion
    completed_milestones = getattr(env, "_completed_milestones", set())

    # Reward for reaching each milestone (based on object/hand state)
    palm_pos = data.xpos[model.body("palm").id].copy()
    hand_to_obj = np.linalg.norm(palm_pos - obj_pos)
    obj_height = obj_pos[2]
    obj_velocity = np.linalg.norm(data.qvel[-6:-3]) if len(data.qvel) > 6 else 0

    # Dynamic milestone rewards
    for milestone_name, milestone_frame in MILESTONES.items():
        if milestone_name in completed_milestones:
            continue

        bonus_key = f"{{milestone_name}}_bonus"
        bonus = REWARD_CONFIG.get(bonus_key, 10.0)

        # Check if milestone achieved based on label
        achieved = False

        if "hold" in milestone_name or "grasp" in milestone_name:
            achieved = n_contacts >= 3 and hand_to_obj < 0.1
        elif "lift" in milestone_name:
            achieved = obj_height > 0.1 and n_contacts >= 2
        elif "wind" in milestone_name or "swing" in milestone_name:
            achieved = obj_height > 0.15 and obj_velocity > 0.5
        elif "release" in milestone_name or "throw" in milestone_name:
            achieved = n_contacts == 0 and obj_velocity > 1.0
        elif "flight" in milestone_name:
            achieved = n_contacts == 0 and obj_height > 0.2
        elif "catch" in milestone_name:
            achieved = n_contacts >= 2 and obj_velocity < 0.3 and getattr(env, "_was_flying", False)
        elif "stabilize" in milestone_name or "stable" in milestone_name:
            achieved = n_contacts >= 3 and obj_velocity < 0.1
        elif "transport" in milestone_name:
            achieved = obj_height > 0.08 and dist_to_target < 0.5
        elif "place" in milestone_name or "drop" in milestone_name:
            achieved = dist_to_target < 0.15
        else:
            # Generic: check if we're past this frame in the trajectory
            achieved = current_frame >= milestone_frame

        if achieved:
            reward += bonus
            completed_milestones.add(milestone_name)

    # Track flying state for catch detection
    if n_contacts == 0 and obj_height > 0.1:
        env._was_flying = True

    env._completed_milestones = completed_milestones

    # Success bonus
    if success:
        reward += REWARD_CONFIG["success_bonus"]

    return reward, {{
        "success": success,
        "obj_height": obj_height,
        "n_contacts": n_contacts,
        "dist_to_target": dist_to_target,
        "completed_milestones": list(completed_milestones)
    }}
'''


def generate_train_content(task_name: str, task_type: str):
    """Generate the train.py content - FULL RESIDUAL RL PIPELINE"""
    return f'''"""
Residual RL Training Script for: {task_name}
Task Type: {task_type}

FULL PIPELINE:
1. Extract hand trajectory from video (MediaPipe)
2. Retarget to robot joint angles
3. Train BC base policy (single-demo with augmentation)
4. Train residual RL on top (TD3 learns corrections)

Modular files:
- config.py: All hyperparameters
- baseline.py: BC policy architecture
- residual.py: Residual RL wrapper
- reward.py: Custom reward function

To run:
  python train.py
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle

# Import from modular files
from config import (
    TASK_NAME, TASK_TYPE, MILESTONES,
    REWARD_CONFIG, BC_CONFIG, RESIDUAL_CONFIG, TRAINING_CONFIG
)
from baseline import BCTrainer, BCConfig, BCBasePolicy
from residual import ResidualRLEnv, ResidualCurriculum
from reward import compute_reward


def extract_trajectory_from_video(video_path: str):
    """
    Extract hand trajectory from video using MediaPipe.

    Returns:
        joint_trajectory: (T, 30) robot joint angles
        grasp_frame: frame where grasp begins
        target_pos: (3,) estimated target position
    """
    from vidreward.extraction.mediapipe_tracker import MediaPipeTracker
    from vidreward.retargeting.landmarks_to_angles import AdroitRetargeter
    from vidreward.utils.video_io import VideoReader

    print(f"Extracting trajectory from: {{video_path}}")

    # 1. Load video frames
    reader = VideoReader(video_path)
    frames = list(reader.read_frames())
    print(f"Loaded {{len(frames)}} frames")

    # 2. Extract hand landmarks with MediaPipe
    tracker = MediaPipeTracker()
    hand_traj = tracker.process_frames(frames)
    tracker.close()
    print(f"Extracted {{len(hand_traj.landmarks)}} hand poses")

    # 3. Retarget to robot joint angles
    retargeter = AdroitRetargeter()
    joint_trajectory = retargeter.retarget_sequence(hand_traj.landmarks)
    joint_trajectory = np.nan_to_num(joint_trajectory, nan=0.0)
    print(f"Retargeted to {{joint_trajectory.shape}} joint trajectory")

    # 4. Load analysis for grasp frame and target
    analysis_path = Path(video_path).parent / "analysis.json"
    grasp_frame = 0
    target_pos = np.array([0.0, 0.0, 0.2])

    if analysis_path.exists():
        import json
        with open(analysis_path) as f:
            analysis = json.load(f)
        grasp_frame = analysis.get("grasp_frame", 0)
        if "target_pos" in analysis:
            target_pos = np.array(analysis["target_pos"])

    return joint_trajectory, grasp_frame, target_pos


def main():
    from stable_baselines3 import TD3
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback

    print("=" * 60)
    print(f"RESIDUAL RL TRAINING: {{TASK_NAME}} ({{TASK_TYPE}})")
    print("=" * 60)

    # Paths
    task_dir = Path("data") / TASK_NAME
    video_path = task_dir / "video.mp4"

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / TASK_NAME / f"{{TASK_NAME}}_{{timestamp}}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {{run_dir}}")

    # =========================================================================
    # SAVE CODE FILES TO RUN DIRECTORY (for reproducibility + LLM context)
    # =========================================================================
    code_dir = run_dir / "code"
    code_dir.mkdir(exist_ok=True)

    # Copy all code files used for this training
    import shutil
    script_dir = Path(__file__).parent
    for code_file in ["config.py", "baseline.py", "residual.py", "reward.py", "train.py"]:
        src = script_dir / code_file
        if src.exists():
            shutil.copy(src, code_dir / code_file)
            print(f"Saved {{code_file}} to run directory")

    # Also save the full config as JSON for easy parsing
    import json
    config_snapshot = {{
        "task_name": TASK_NAME,
        "task_type": TASK_TYPE,
        "milestones": MILESTONES,
        "reward_config": REWARD_CONFIG,
        "bc_config": BC_CONFIG,
        "residual_config": RESIDUAL_CONFIG,
        "training_config": TRAINING_CONFIG,
        "timestamp": timestamp,
    }}
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2, default=str)
    print("Saved config.json snapshot")

    # =========================================================================
    # STEP 1: Extract trajectory from video
    # =========================================================================
    print("\\n[Step 1/4] Extracting trajectory from video...")

    joint_traj, grasp_frame, target_pos = extract_trajectory_from_video(str(video_path))

    # Save trajectory
    with open(run_dir / "joint_trajectory.pkl", "wb") as f:
        pickle.dump({{"trajectory": joint_traj, "grasp_frame": grasp_frame, "target_pos": target_pos}}, f)

    # =========================================================================
    # STEP 2: Train BC base policy
    # =========================================================================
    print("\\n[Step 2/4] Training BC base policy...")

    # Create env for data collection
    env = gym.make("AdroitHandRelocate-v1")

    # Collect state-action pairs by replaying trajectory in sim
    bc_config = BCConfig()
    trainer = BCTrainer(bc_config)
    states, actions = trainer.collect_data_from_sim(env, joint_traj)
    print(f"Collected {{len(states)}} state-action pairs")

    # Train BC policy
    bc_policy = trainer.train(states, actions, verbose=True)

    # Save BC policy
    import torch
    torch.save(bc_policy.state_dict(), run_dir / "bc_policy.pt")
    print(f"Saved BC policy to {{run_dir / 'bc_policy.pt'}}")

    env.close()

    # =========================================================================
    # STEP 3: Create Residual RL environment
    # =========================================================================
    print("\\n[Step 3/4] Setting up Residual RL environment...")

    base_env = gym.make("AdroitHandRelocate-v1")

    # Wrap with residual RL
    env = ResidualRLEnv(
        env=base_env,
        base_policy=bc_policy,
        residual_scale=RESIDUAL_CONFIG["initial_scale"],
        residual_penalty_weight=RESIDUAL_CONFIG["penalty_weight"],
    )
    env = Monitor(env, str(run_dir))

    # Curriculum callback
    curriculum = ResidualCurriculum(
        env.env if hasattr(env, 'env') else env,  # Unwrap Monitor
        initial_scale=RESIDUAL_CONFIG["initial_scale"],
        final_scale=RESIDUAL_CONFIG["final_scale"],
        warmup_steps=RESIDUAL_CONFIG["warmup_steps"],
    )

    # Custom callback for curriculum + logging
    class CurriculumCallback(BaseCallback):
        def __init__(self, curriculum):
            super().__init__()
            self.curriculum = curriculum

        def _on_step(self):
            metrics = self.curriculum.on_step()
            for k, v in metrics.items():
                self.logger.record(f"curriculum/{{k}}", v)
            return True

    # =========================================================================
    # STEP 4: Train TD3 residual policy
    # =========================================================================
    print("\\n[Step 4/4] Training TD3 residual policy...")

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=TRAINING_CONFIG["action_noise_sigma"] * np.ones(n_actions)
    )

    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        buffer_size=TRAINING_CONFIG["buffer_size"],
        learning_starts=TRAINING_CONFIG["learning_starts"],
        batch_size=TRAINING_CONFIG["batch_size"],
        tau=TRAINING_CONFIG.get("tau", 0.005),
        gamma=TRAINING_CONFIG.get("gamma", 0.99),
        action_noise=action_noise,
        policy_kwargs=dict(net_arch=TRAINING_CONFIG["policy_arch"]),
        tensorboard_log=str(run_dir / "tb"),
        verbose=1
    )

    print(f"\\nStarting TD3 training for {{TRAINING_CONFIG['total_timesteps']}} steps...")
    print(f"Residual scale: {{RESIDUAL_CONFIG['initial_scale']}} -> {{RESIDUAL_CONFIG['final_scale']}}")

    model.learn(
        total_timesteps=TRAINING_CONFIG["total_timesteps"],
        callback=CurriculumCallback(curriculum),
        progress_bar=True
    )

    # Save final model
    model.save(run_dir / "td3_residual_final")
    print(f"\\nSaved TD3 model to {{run_dir / 'td3_residual_final.zip'}}")

    # =========================================================================
    # Done!
    # =========================================================================
    print("\\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Results saved to: {{run_dir}}")
    print(f"  - bc_policy.pt: BC base policy")
    print(f"  - td3_residual_final.zip: Trained residual policy")
    print(f"  - tb/: TensorBoard logs")

    env.close()


if __name__ == "__main__":
    main()
'''


@app.get("/api/experiment/script/generate/{task_name}")
def generate_training_script(task_name: str):
    """
    Generate modular training scripts for FULL residual RL pipeline:
    - config.py: All hyperparameters
    - baseline.py: BC policy (behavioral cloning from demo)
    - residual.py: Residual RL wrapper (action = base + residual)
    - reward.py: Custom milestone-based reward
    - train.py: Full training pipeline
    """
    task_dir = Path("data") / task_name
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail="Task not found")

    analysis_path = task_dir / "analysis.json"
    if not analysis_path.exists():
        raise HTTPException(status_code=400, detail="Task not analyzed yet")

    with open(analysis_path) as f:
        analysis = json.load(f)

    task_type = analysis.get("task_type", "pick")
    milestones = analysis.get("milestones", [])

    # Generate ALL modular content
    config_content = generate_config_content(task_name, task_type, milestones, analysis)
    baseline_content = generate_baseline_content(task_name, task_type)
    residual_content = generate_residual_content(task_name, task_type)
    reward_content = generate_reward_content(task_name, task_type, milestones)
    train_content = generate_train_content(task_name, task_type)

    # Build reward config for response
    default_config = {
        "time_penalty": -0.1,
        "contact_scale": 0.1,
        "success_bonus": 500.0,
    }
    for m in milestones:
        label = m.get("label", "unknown").lower().replace(" ", "_")
        default_config[f"{label}_bonus"] = 10.0
    if "reward_config" in analysis:
        default_config.update(analysis["reward_config"])

    return {
        "config_code": config_content,
        "baseline_code": baseline_content,
        "residual_code": residual_content,
        "reward_code": reward_content,
        "train_code": train_content,
        "content": train_content,  # Keep for backwards compatibility
        "task_name": task_name,
        "task_type": task_type,
        "reward_config": default_config,
        "milestones": milestones
    }



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


@app.post("/api/runs/upload")
async def upload_run(file: UploadFile = File(...)):
    """
    Upload a training run zip file.
    Expected structure: run_name.zip containing config.pkl, plots/, checkpoints/, etc.
    """
    import zipfile
    import io

    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only .zip files are accepted")

    # Max 500MB for run data
    MAX_SIZE = 500 * 1024 * 1024

    # Read file into memory
    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="File too large (Max 500MB)")

    # Extract run name from filename (e.g., "pick-3_20240209_123456.zip" -> "pick-3_20240209_123456")
    run_name = Path(file.filename).stem

    # Determine group name (first part before underscore with date)
    # e.g., "pick-3_20240209_123456" -> group is "pick-3" or infer from content
    parts = run_name.split('_')
    if len(parts) >= 2:
        # Assume format: taskname_timestamp or taskname_date_time
        group_name = parts[0]
    else:
        group_name = "uploaded"

    # Create run directory
    run_dir = RUNS_DIR / group_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Extract zip
    try:
        with zipfile.ZipFile(io.BytesIO(content), 'r') as zf:
            # Check for required files
            file_list = zf.namelist()
            print(f"[Upload] Extracting {len(file_list)} files to {run_dir}")

            for member in zf.namelist():
                # Skip directories
                if member.endswith('/'):
                    continue

                # Handle nested structure - if files are in a subfolder, flatten
                # e.g., "run_name/config.pkl" -> "config.pkl"
                member_path = Path(member)
                if len(member_path.parts) > 1 and member_path.parts[0] == run_name:
                    # Remove the run_name prefix
                    target_name = str(Path(*member_path.parts[1:]))
                else:
                    target_name = member

                target_path = run_dir / target_name
                target_path.parent.mkdir(parents=True, exist_ok=True)

                with zf.open(member) as src, open(target_path, 'wb') as dst:
                    dst.write(src.read())

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid zip file")

    # Upload to S3 if enabled
    if storage.enabled:
        try:
            print(f"[Upload] Syncing {run_name} to S3...")
            storage.upload_dir(str(run_dir), f"runs/{group_name}/{run_name}")
            print(f"[Upload] S3 sync complete")
        except Exception as e:
            print(f"[Upload] S3 sync failed: {e}")

    # Check what we got
    has_config_pkl = (run_dir / "config.pkl").exists()
    has_config_json = (run_dir / "config.json").exists()
    has_history = (run_dir / "plots" / "history.pkl").exists()
    has_model = (
        (run_dir / "td3_residual_final.zip").exists() or
        (run_dir / "td3_final.zip").exists() or
        len(list(run_dir.glob("checkpoints/*.zip"))) > 0
    )
    has_bc_policy = (run_dir / "bc_policy.pt").exists()

    # Check for code directory (critical for LLM context!)
    code_dir = run_dir / "code"
    has_code = code_dir.exists()
    code_files = []
    if has_code:
        code_files = [f.name for f in code_dir.glob("*.py")]

    return {
        "status": "uploaded",
        "run_name": run_name,
        "group": group_name,
        "path": str(run_dir),
        "has_config": has_config_pkl or has_config_json,
        "has_history": has_history,
        "has_model": has_model,
        "has_bc_policy": has_bc_policy,
        "has_code": has_code,
        "code_files": code_files,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
