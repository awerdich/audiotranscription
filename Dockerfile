FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS base

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG DEV_audiotranscription

ENV \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_SRC=/src \
    NO_COLOR=true \
    UV_COMPILE_BYTECODE=1 \
    UV_SYSTEM_PYTHON=true \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON_PREFERENCE=only-system \
    UV_LINK_MODE=copy \
    UV_TOOL_BIN_DIR=/usr/bin \
    UV_PROJECT_ENVIRONMENT=/usr

# Set non-interactive frontend for apt
ENV DEBIAN_FRONTEND=noninteractive

# Make sure "python" refers to the system Python
RUN ln -sf /usr/bin/python3 /usr/local/bin/python

RUN apt-get -y update && \
    apt-get -y install \ 
    libgl1 \
    ffmpeg

# Ports for jupyter
EXPOSE 8888

RUN mkdir -p /app
WORKDIR /app

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# Install the project
COPY src/audiotranscription/__init__.py \
    src/audiotranscription/VERSION src/audiotranscription
COPY pyproject.toml uv.lock ./
RUN uv sync --locked

# Copy bash scripts and set executable flags
RUN mkdir -p /run_scripts
COPY /bash_scripts/* /run_scripts
RUN chmod +x /run_scripts/*

# Run the jupyter server
CMD ["/bin/bash", "/run_scripts/docker_entry"]