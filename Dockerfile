# llmlocal — CPU-only Linux image (llama.cpp backend + FastAPI API server).
# llama-server comes from the pinned llama.cpp release below (same build as
# the native Windows runtime); Python deps (FastAPI/uvicorn) come from
# requirements.txt. The GGUF model is mounted at runtime, never baked in.
FROM ubuntu:24.04

# Pinned llama.cpp release (verify checksum when bumping:
# https://github.com/ggml-org/llama.cpp/releases/tag/b11438)
ARG LLAMA_BUILD=b11438
ARG LLAMA_SHA256=afd4262e6f41b3c9d7b41605c4e080969f8343a12a72a5c1c732f59555c92a1e

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip curl ca-certificates tar libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Linux CPU llama-server lands at tools/llama.cpp/llama-server — the exact
# path backend/llama_cpp.py::_find_server_exe() probes on non-Windows hosts,
# so no application code changes are needed. The --version call is a
# build-time smoke test: it fails the image build on missing shared libraries
# or a glibc mismatch.
RUN mkdir -p /app/tools/llama.cpp \
    && curl -fsSL -o /tmp/llama.tar.gz \
        "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_BUILD}/llama-${LLAMA_BUILD}-bin-ubuntu-x64.tar.gz" \
    && echo "${LLAMA_SHA256}  /tmp/llama.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/llama.tar.gz -C /app/tools/llama.cpp --strip-components=1 \
    && rm /tmp/llama.tar.gz \
    && /app/tools/llama.cpp/llama-server --version

WORKDIR /app

# Python dependencies first, in their own layer for caching. ubuntu:24.04
# marks its system Python externally managed (PEP 668); --break-system-packages
# installs into the image's system site-packages, which is appropriate for an
# application-only container (no apt Python packages are touched).
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /app/requirements.txt

COPY server.py /app/server.py
COPY backend/ /app/backend/
COPY app/ /app/app/

# Runtime configuration defaults. compose overrides most of these; nothing
# secret lives in the image (no HF_TOKEN: the mounted GGUF needs no download).
# LLM_PORT is deliberately unset: the in-container API port is fixed at 8000.
ENV LLM_BACKEND=llama_cpp \
    LLM_MODEL=Qwen3-1.7B-Q4_K_M \
    LLM_MODEL_PATH=models/gguf/Qwen3-1.7B-Q4_K_M.gguf \
    LLM_BACKEND_PORT=8100 \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["python3", "/app/server.py"]
