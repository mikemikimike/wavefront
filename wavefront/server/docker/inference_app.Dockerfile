# CPU-only: torch comes from the pytorch-cpu index (see pyproject.toml), so
# the CUDA base image would only add unused GPU libraries.
FROM ubuntu:22.04

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.12.24 /uv /uvx /bin/

# Keep uv/pip download caches out of the image layers
ENV UV_NO_CACHE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    ca-certificates \
    libpq-dev \
    gcc \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create user early so uv sync runs as appuser — avoids Python binary landing in /root/.local
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

COPY --chown=appuser:appuser wavefront/server/pyproject.toml wavefront/server/uv.lock wavefront/server/.python-version ./
COPY --chown=appuser:appuser wavefront/server/modules/common_module /app/modules/common_module
COPY --chown=appuser:appuser wavefront/server/packages/flo_cloud /app/packages/flo_cloud
COPY --chown=appuser:appuser wavefront/server/packages/flo_lib /app/packages/flo_lib
COPY --chown=appuser:appuser wavefront/server/apps/inference_app /app/apps/inference_app

USER appuser

RUN uv sync --package inference-app --frozen --no-dev

# The app runs from the venv; drop pip and ensurepip's bundled wheels from the
# uv-managed interpreter (they vendor their own urllib3, msgpack, setuptools)
RUN find "$(uv python dir)" -depth \( -path '*/site-packages/pip' -o -path '*/site-packages/pip-*' -o -path '*/ensurepip/_bundled' \) -exec rm -rf {} +

WORKDIR /app/apps/inference_app/inference_app

CMD ["uv", "run", "--frozen", "--no-sync", "server.py"]
