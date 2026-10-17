FROM python:3.11-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.12.24 /uv /uvx /bin/

# Keep uv/pip download caches out of the image layers
ENV UV_NO_CACHE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# The app runs from the uv venv, so drop the base image's system pip/setuptools/wheel
# (and the vendored copies of urllib3, msgpack, etc. they carry)
RUN pip uninstall -y pip setuptools wheel && \
    rm -rf /usr/local/lib/python3.11/ensurepip/_bundled

COPY wavefront/server/pyproject.toml wavefront/server/uv.lock ./

COPY wavefront/server/modules/common_module /app/modules/common_module

COPY wavefront/server/packages/flo_cloud /app/packages/flo_cloud
COPY wavefront/server/packages/flo_lib /app/packages/flo_lib

COPY wavefront/server/apps/floconsole /app/apps/floconsole

COPY wavefront/server/scripts/console-server-init.sh /app/scripts/console-server-init.sh
RUN chmod +x /app/scripts/console-server-init.sh

RUN uv sync --package floconsole --frozen --no-dev

# uv.lock covers the whole workspace, but only this app's deps are in the venv;
# scanners (AWS Inspector) report every package in it. `uv run --frozen --no-sync`
# does not need it.
RUN rm uv.lock

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to the non-root user for all subsequent instructions
USER appuser

ENTRYPOINT ["/app/scripts/console-server-init.sh"]
