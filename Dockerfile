# syntax=docker/dockerfile:1.4
# Lightweight production API server - no ML dependencies
FROM python:3.11-slim

WORKDIR /app

# Install deps FIRST (cached unless requirements-prod.txt changes)
COPY requirements-prod.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-prod.txt

# Copy only server code (see .dockerignore)
COPY . .

EXPOSE 8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8000"]
