import os
import pickle
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

app = FastAPI(title="Gemini Robotics Iteration Dashboard API")

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
    for group_dir in RUNS_DIR.iterdir():
        if group_dir.is_dir():
            for run_dir in group_dir.iterdir():
                if run_dir.is_dir() and (run_dir / "config.pkl").exists():
                    try:
                        with open(run_dir / "config.pkl", "rb") as f:
                            config = pickle.load(f)
                        
                        # Clean config for JSON (numpy arrays to lists)
                        clean_config = {}
                        for k, v in config.items():
                            if hasattr(v, "tolist"):
                                clean_config[k] = v.tolist()
                            else:
                                clean_config[k] = v

                        has_history = (run_dir / "plots" / "history.pkl").exists()
                        
                        # Check for videos in the run dir or checkpoints
                        videos = list(run_dir.glob("*.mp4"))
                        has_video = len(videos) > 0
                        
                        all_runs.append(RunInfo(
                            id=run_dir.name,
                            group=group_dir.name,
                            timestamp=run_dir.name.split("_")[-1] if "_" in run_dir.name else "",
                            config=clean_config,
                            has_history=has_history,
                            has_video=has_video
                        ))
                    except Exception as e:
                        print(f"Error loading run {run_dir}: {e}")
    return all_runs

@app.get("/api/run/{group}/{run_id}/history")
def get_run_history(group: str, run_id: str):
    history_path = RUNS_DIR / group / run_id / "plots" / "history.pkl"
    if not history_path.exists():
        raise HTTPException(status_code=404, detail="History not found")
    
    with open(history_path, "rb") as f:
        history = pickle.load(f)
    
    # Clean history for JSON
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
