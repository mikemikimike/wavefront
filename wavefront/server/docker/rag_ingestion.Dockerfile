# bookworm: buster (Debian 10) is end-of-life and no longer receives security fixes
FROM python:3.11-slim-bookworm

# Copy UV from official image
COPY --from=ghcr.io/astral-sh/uv:0.12.24 /uv /uvx /bin/

# Keep uv/pip download caches out of the image layers
ENV UV_NO_CACHE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# The app runs from the uv venv, so drop the base image's system pip/setuptools/wheel
# (and the vendored copies of urllib3, msgpack, etc. they carry)
RUN pip uninstall -y pip setuptools wheel && \
    rm -rf /usr/local/lib/python3.11/ensurepip/_bundled

# Set working directory
WORKDIR /app

# Copy project files
COPY wavefront/server/pyproject.toml wavefront/server/uv.lock ./
COPY wavefront/server/background_jobs/rag_ingestion ./background_jobs/rag_ingestion/
COPY wavefront/server/packages/flo_cloud ./packages/flo_cloud/
COPY wavefront/server/packages/flo_utils ./packages/flo_utils/
COPY wavefront/server/packages/flo_lib ./packages/flo_lib/
COPY wavefront/server/packages/flo_stream ./packages/flo_stream/
COPY wavefront/server/modules/db_repo_module ./modules/db_repo_module/
COPY wavefront/server/modules/common_module ./modules/common_module/
COPY wavefront/server/scripts/rag_ingestion/startup-rag-ingestion.sh ./background_jobs/rag_ingestion/

# Install dependencies
RUN uv sync --package rag-ingestion --frozen --no-dev

# Download the tiktoken encoding file and NLTK data
RUN mkdir -p /root/.cache/tiktoken
RUN /app/.venv/bin/python -c "import tiktoken; enc = tiktoken.encoding_for_model('gpt-4')"
RUN /app/.venv/bin/python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

WORKDIR /app/background_jobs/rag_ingestion

# Make startup script executable
RUN chmod +x startup-rag-ingestion.sh

# Create a non-root user with UID 1000 and ensure files are owned by that user
# This allows Kubernetes pods using securityContext.runAsUser: 1000 to work
RUN set -eux \
    && groupadd -g 1000 appuser || true \
    && useradd -u 1000 -g 1000 -m -s /usr/sbin/nologin appuser || true \
    && chown -R 1000:1000 /app /root/.cache/tiktoken || true

# Switch to the non-root user (UID 1000) so containers can run with securityContext 1000
USER 1000

# Set entrypoint to run startup script
CMD ["./startup-rag-ingestion.sh"]
