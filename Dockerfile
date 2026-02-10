# Server build with video analysis (no RL/training deps)
FROM python:3.11-slim

# System deps for video processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (layer cached if requirements unchanged)
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# Copy app code
COPY . .

EXPOSE 8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8000"]
