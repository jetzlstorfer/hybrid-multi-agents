# Backend image: FastAPI + AG-UI server + edge/cloud agents.
# foundry-local-sdk is intentionally NOT installed in this image — Foundry
# Local runs on the MicroShift host and is reached via an ExternalName Service.
# The cluster image therefore stays small and CPU-only.
FROM python:3.12-slim AS build

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /build

COPY pyproject.toml ./
COPY src ./src
RUN pip install --upgrade pip build
RUN pip wheel -w /wheels .

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN useradd --uid 1001 --create-home --shell /usr/sbin/nologin app
WORKDIR /app

COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# models.yaml is NOT baked into the image — it is injected at runtime via a
# Kubernetes ConfigMap mounted at /config/models.yaml.  The HYBRID_DEMO_MODELS_FILE
# env var in the Deployment manifest points the app at that path.
COPY samples /app/samples

USER 1001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD \
  python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=2)" || exit 1

ENTRYPOINT ["uvicorn", "hybrid_demo.ag_ui_server:app", "--host", "0.0.0.0", "--port", "8000"]
