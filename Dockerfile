# Base-image registry prefix. Empty default = public Docker Hub; a private
# deploy mirror passes --build-arg BASE=registry.example/library/ .
ARG BASE=
# Per-item GPU video-prep worker.
#
# Base image: nvidia/cuda:12.3.2-runtime-ubuntu22.04. We need the
# userspace CUDA libraries (libcuda.so.1 ships from the host via the
# device plugin) and a recent libnvidia-encode-API, but NOT the full
# CUDA dev toolkit — we don't compile anything CUDA-aware. The runtime
# tag is ~1.5 GB; the devel tag is ~5 GB and would only buy us nvcc.
#
# ffmpeg is pulled in as a prebuilt NVENC-enabled static build, because
# Ubuntu 22.04's stock package was built without --enable-nvenc — feeding
# it `-c:v hevc_nvenc` fails with "Unknown encoder 'hevc_nvenc'". The
# static build ships NVENC linked in plus matching libavcodec/libavformat,
# so the encoder flags below behave identically regardless of host ffmpeg.
FROM ${BASE}nvidia/cuda:12.3.2-runtime-ubuntu22.04 AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    # Tell the NVIDIA container runtime which capabilities the
    # container needs. `video` is what unlocks NVENC; `compute` is
    # what unlocks CUDA. Both required for hevc_nvenc.
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,video,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-venv \
      ca-certificates \
      curl \
      xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Install a prebuilt NVENC-enabled ffmpeg static build. The tarball is
# fetched and unpacked at build-time only — no runtime download traffic.
# The `ffmpeg`/`ffprobe` binaries land in /usr/local/bin so the Python
# worker can call them via bare PATH lookups.
#
# PIN to the ffmpeg 7.1 release branch — do NOT track `master-latest`.
# master drifts to bleeding-edge nvenc SDKs: a mid-2026 master build began
# requiring nvenc API 13.1 (Nvidia driver >= 610) and aborted every encode
# with "Driver does not support the required nvenc API version. Required:
# 13.1 Found: 13.0" on our GPU nodes (RTX 3090, driver 580 / nvenc 13.0).
# The n7.1 branch links an nvenc SDK compatible with driver >= ~550 and only
# takes bugfix backports, so it never bumps the driver floor out from under
# the cluster. Bump this to a newer branch ONLY after the GPU nodes' driver
# is verified new enough (nvidia-smi on the worker) and the encode re-validated.
# Override with --build-arg FFMPEG_BUILD_URL=... to pin a mirror.
ARG FFMPEG_BUILD_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-linux64-gpl-7.1.tar.xz
RUN curl -fsSL "${FFMPEG_BUILD_URL}" -o /tmp/ffmpeg.tar.xz \
    && mkdir -p /tmp/ffmpeg \
    && tar -xJf /tmp/ffmpeg.tar.xz -C /tmp/ffmpeg --strip-components=1 \
    && install -m 0755 /tmp/ffmpeg/bin/ffmpeg /usr/local/bin/ffmpeg \
    && install -m 0755 /tmp/ffmpeg/bin/ffprobe /usr/local/bin/ffprobe \
    && rm -rf /tmp/ffmpeg /tmp/ffmpeg.tar.xz

WORKDIR /app

# Bump pip + setuptools + wheel. Ubuntu 22.04 ships setuptools 59,
# which silently produces a bogus `UNKNOWN-0.0.0` wheel from a pyproject
# that uses `[tool.setuptools.packages.find]` (that table form needs
# setuptools >= 61). Without this bump the `pip install .` below "works"
# but installs an empty package, and `python3 -m transcoder.main` then
# fails with ModuleNotFoundError at pod start.
RUN pip3 install --upgrade pip setuptools wheel

# Runtime deps first (cached separately from source). confluent-kafka's
# manylinux wheel bundles librdkafka, so no apt package is needed for the
# Kafka client on this glibc (cuda/ubuntu) base.
RUN pip3 install \
      confluent-kafka==2.5.3 \
      httpx==0.28.1 structlog==25.4.0 fastapi==0.118.0 "uvicorn[standard]==0.32.0"

COPY pyproject.toml ./
COPY src ./src
# Non-editable install. The Ubuntu 22.04 setuptools (~59.x) predates
# PEP 660 ('build_editable' hook), so `pip install -e .` fails with
# "build backend is missing the build_editable hook". A regular
# install copies src/transcoder into site-packages, which is what we
# want for the production image anyway — editable mode was only
# useful when this Dockerfile was being iterated on locally.
RUN pip3 install --no-deps .

# OpenShift runs containers under a random non-root uid that's in GID 0.
# Make everything group-writable so the runtime can own it.
RUN chown -R 0:0 /app && chmod -R g=u /app

EXPOSE 8080
CMD ["python3", "-m", "transcoder.main"]
