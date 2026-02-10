# OVERFIT: One Video Episode Reward Function from Imitation Tracking

OVERFIT is an end-to-end pipeline for teaching complex robotic tasks to an Adroit Shadow Hand using a single real-world video demonstration. It leverages **Gemini 3.0** for video analysis, task understanding, and automated reward tuning, combined with **Stable Baselines3 (TD3)** for Residual Reinforcement Learning.

> **Note:** This project is best run locally for full functionality (training, video analysis, evaluation). The cloud-hosted web app is for **demonstration purposes only** - it provides read-only access to experiment data stored in S3 but cannot run training or analysis scripts.

## Features

- **Single-Video Learning**: Extract task structure from one demonstration video
- **Gemini-Powered Analysis**: Automatic milestone detection, grasp frame identification, and failure mode analysis
- **Interactive Dashboard**: Real-time experiment monitoring with video labeling and chat interface
- **Cloud-Hybrid Architecture**: Train locally, monitor remotely via S3-compatible storage
- **Automated Reward Tuning**: Self-improving reward functions through Gemini critique loops

---

## Quick Start

### 1. Installation

Requires Python 3.10+ with CUDA support recommended for training.

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Required - Gemini API for video analysis and critiques
GEMINI_API_KEY=your-gemini-api-key

# Optional - Cloud storage for remote dashboard access
BLOB_BUCKET_NAME=your-bucket-name
BLOB_ACCESS_KEY=your-access-key
BLOB_SECRET_KEY=your-secret-key
BLOB_ENDPOINT_URL=https://your-endpoint.com  # For R2, Spaces, MinIO
```

### 3. Launch the Dashboard

**Terminal 1: FastAPI Backend**
```bash
python backend/server.py
```

**Terminal 2: React Frontend**
```bash
cd dashboard
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) to access the dashboard.

---

## Training

### Basic Training
Start a training session with a demonstration video:
```bash
python scripts/train_grasp_residual.py --video data/pick-3.mp4 --timesteps 500000
```

### Reward Versions
OVERFIT includes multiple reward configurations:
```bash
# v1: Original basic reward
# v2: High-reward/Unstable
# v3: Stabilized expert heuristics (Default)
python scripts/train_grasp_residual.py --video data/pick-3.mp4 --reward-version v3
```

### Custom Reward Logic
Use an external reward file:
```bash
python scripts/train_grasp_residual.py --video data/pick-3.mp4 --reward-file experiments/v4_custom.py
```

### Automated Iteration (Gemini RL Tuner)
Automatically improve reward functions by analyzing training curves:
```bash
python scripts/gemini_rl_tuner.py --run-dir runs/residual_pick3/pick-3_...
```

---

## Project Structure

```
├── backend/              # FastAPI server
│   └── server.py         # API endpoints, Gemini chat, video serving
├── dashboard/            # React 19 + Vite + Tailwind frontend
│   └── src/components/   # UI components (ExperimentHub, VideosView, etc.)
├── scripts/              # Core execution scripts
│   ├── train_grasp_residual.py   # Main RL training loop
│   ├── train_cli.py              # CLI training interface
│   ├── gemini_rl_tuner.py        # Automated reward improvement
│   ├── gemini_video_analyzer.py  # Video analysis pipeline
│   ├── gemini_sim_critique.py    # Simulation critique system
│   ├── label_video.py            # Video labeling utilities
│   └── reward_registry.py        # Named reward versions (v1, v2, v3)
├── vidreward/            # Core library
│   ├── extraction/       # Video feature extraction
│   ├── retargeting/      # Motion retargeting to robot
│   ├── rewards/          # Reward function implementations
│   ├── phases/           # Phase-aware reward logic
│   └── utils/            # Storage, helpers
├── data/                 # Videos, trajectories, analysis results
├── runs/                 # Training artifacts (models, plots, history)
└── outputs/              # Generated outputs
```

---

## Dashboard Features

- **Experiment Hub**: Browse and compare training runs
- **Videos View**: Upload and manage demonstration videos
- **Analysis Review**: Inspect Gemini's video analysis with milestone markers
- **Experiment Design**: Configure new training experiments
- **Live Charts**: Real-time reward and success rate visualization
- **Chat Interface**: Interactive labeling with Gemini assistance

---

## Cloud-Hybrid Mode

OVERFIT supports a distributed workflow using S3-compatible storage (AWS S3, Cloudflare R2, DigitalOcean Spaces, MinIO):

1. **Local Training**: Run GPU-intensive RL training on your local machine. The training script uploads checkpoints and plots to your configured bucket.
2. **Remote Dashboard**: Deploy the dashboard on a cloud server with the same S3 credentials. It fetches run data from the bucket on-demand.

This lets you monitor training progress from anywhere without exposing your local machine.

---

## Docker

```bash
docker-compose up --build
```

---

## Tech Stack

- **RL**: Stable Baselines3, MuJoCo, Gymnasium-Robotics
- **AI**: Gemini 2.0 (google-genai), PyTorch, MediaPipe, OpenCLIP
- **Backend**: FastAPI, Uvicorn, Pydantic
- **Frontend**: React 19, Vite, Tailwind CSS, Recharts
- **Storage**: Boto3 (S3-compatible)
