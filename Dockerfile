# syntax=docker/dockerfile:1.4
# Server build with video analysis (no RL/training deps)
FROM python:3.11-slim

# System deps for video processing (ffmpeg, opencv)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (cached unless requirements-server.txt changes)
COPY requirements-server.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-server.txt

# Copy everything
COPY . .

EXPOSE 8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8000"]
