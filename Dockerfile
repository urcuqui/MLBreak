# --- Stage 1: build the React UI -------------------------------------------
FROM node:20.18-alpine AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# --- Stage 2: Python runtime -----------------------------------------------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MLBREAK_DATA_DIR=/data
WORKDIR /app
# Hash-pinned runtime dependencies first (reproducible numerical stack, cached layer), then the project
# itself without re-resolving dependencies.
COPY requirements/runtime-py3.12.txt requirements/
RUN pip install --no-cache-dir --require-hashes -r requirements/runtime-py3.12.txt
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir --no-deps . && useradd --create-home --uid 10001 mlbreak && mkdir -p /data && chown mlbreak /data
COPY security_cases/ security_cases/
COPY --from=ui /ui/dist frontend/dist
USER mlbreak
EXPOSE 8000
# 0.0.0.0 is required inside the container; compose publishes it on 127.0.0.1 only.
CMD ["python", "-c", "import uvicorn; from mlbreak.api import create_app; from pathlib import Path; uvicorn.run(create_app(cases=Path('/app/security_cases'), frontend_dist=Path('/app/frontend/dist')), host='0.0.0.0', port=8000)"]
