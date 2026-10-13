# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

FROM builder AS test-builder
RUN uv sync --frozen

FROM python:3.12-slim AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --create-home app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/sp35 /app/sp35
COPY --from=builder --chown=app:app /app/ingest /app/ingest
COPY --from=builder --chown=app:app /app/vision /app/vision
COPY --from=builder --chown=app:app /app/kb /app/kb
COPY --from=builder --chown=app:app /app/tools /app/tools

USER app

# one-shot check (exits when done), so no HEALTHCHECK
CMD ["python", "-m", "tools.sp35_reference_check"]

FROM runtime AS test
# Hypothesis' example database must live on the writable /tmp (the container is read-only)
ENV HYPOTHESIS_STORAGE_DIRECTORY=/tmp/hypothesis
COPY --from=test-builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/tests /app/tests
COPY --from=builder --chown=app:app /app/app/backend /app/app/backend
COPY --from=builder --chown=app:app /app/pyproject.toml /app/pyproject.toml
CMD ["/app/.venv/bin/pytest", "-o", "cache_dir=/tmp/pytest-cache"]

# GPU target for the Linux CUDA worker (RTX 3060 / 5070 Ti). Same pins as the
# Windows .venv-win: torch/torchvision +cu128 and the rest have Linux wheels;
# triton-windows is skipped by its platform marker. Ubuntu 24.04 ships Python 3.12
# (22.04 does not). Weights are read from the mounted ./data volume.
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04 AS gpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y python3.12 python3.12-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip

COPY requirements/vision-win.txt ./requirements-vision.txt
# pydantic is the one project dependency (sp35/kb.py) the vision pins do not include
RUN pip install --no-cache-dir -r requirements-vision.txt "pydantic>=2"

COPY kb ./kb
COPY vision ./vision
COPY ingest ./ingest
COPY sp35 ./sp35
COPY tools ./tools

RUN groupadd --system app && useradd --system --gid app --create-home app
USER app

CMD ["python", "-m", "vision.infer", "--help"]
