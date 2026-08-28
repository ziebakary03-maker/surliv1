# syntax=docker/dockerfile:1

# ---------- Stage 1: build frontend ----------
FROM node:20-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ---------- Stage 2: backend + runtime ----------
FROM python:3.11-slim AS runtime

# Dépendances système : OpenCV a besoin de libgl/libglib, ffmpeg pour
# l'encodage vidéo robuste (fallback si le codec OpenCV pose problème).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend ./backend
COPY worker ./worker
COPY scripts ./scripts

# Frontend buildé, servi statiquement (ou via un reverse proxy en prod)
COPY --from=frontend-build /frontend/dist ./frontend/dist

ENV PYTHONPATH=/app/backend
ENV LOCAL_STORAGE_PATH=/data/storage

RUN mkdir -p /data/storage

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Par défaut, lance le serveur web. Le process "worker" est démarré
# séparément (voir docker-compose.yml et fly.toml [processes]).
CMD ["uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
