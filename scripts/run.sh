#!/bin/bash
set -e

# Always work from the repo root (this script lives in scripts/).
cd "$(dirname "$0")/.."

# Load env variables
if [ -f .env ]; then
    # Read .env file and export variables, ignoring comments and empty lines
    while IFS= read -r line || [ -n "$line" ]; do
        # Remove Windows carriage return if present
        line=${line%$'\r'}

        # Skip empty lines and comments
        if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
            continue
        fi

        # Export the variable if it contains an equals sign.
        # Real environment variables win (same rule as server.py's .env loader).
        if [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*(.*)$ ]]; then
            key="${BASH_REMATCH[1]}"
            if [ -z "${!key+x}" ]; then
                export "$key=${BASH_REMATCH[2]}"
            fi
        fi
    done < .env

    # Debug: Show loaded config (never print secrets)
    echo "📋 Loaded .env: LLM_BACKEND=${LLM_BACKEND:-llama_cpp} model=${LLM_MODEL:-<default>} port=${LLM_PORT:-${VLLM_PORT:-8000}}"
fi

BACKEND=${LLM_BACKEND:-llama_cpp}
PORT=${LLM_PORT:-${VLLM_PORT:-8000}}
MODE="native"
FORCE_BUILD=0
for arg in "$@"; do
    case "$arg" in
        --docker) MODE="docker" ;;
        --force-build|-f) FORCE_BUILD=1 ;;
        -h|--help)
            echo "Usage: ./scripts/run.sh [--docker] [--force-build]"
            echo "  (default)     native run: python server.py (LLM_BACKEND=llama_cpp|transformers)"
            echo "  --docker      container path via docker compose (default service llama_cpp; legacy: transformers/vllm profiles)"
            echo "  --force-build rebuild the Docker image (only with --docker)"
            exit 0
            ;;
    esac
done

if [ "$MODE" = "native" ]; then
    echo "🚀 Starting llmlocal natively (backend=$BACKEND, port=$PORT)"
    if ! command -v python >/dev/null 2>&1; then
        echo "❌ python not found on PATH (install Python 3.10+)"
        exit 1
    fi
    if [ "$BACKEND" = "llama_cpp" ]; then
        MODEL_PATH=${LLM_MODEL_PATH:-models/gguf/Qwen3-1.7B-Q4_K_M.gguf}
        if [ ! -f "tools/llama.cpp/llama-server.exe" ] && ! command -v llama-server >/dev/null 2>&1; then
            echo "❌ llama-server not found."
            echo "   Download the Windows CPU release (llama-b*-bin-win-cpu-x64.zip) from"
            echo "   https://github.com/ggml-org/llama.cpp/releases and extract it into tools/llama.cpp/"
            exit 1
        fi
        if [ ! -f "$MODEL_PATH" ]; then
            echo "❌ GGUF model not found: $MODEL_PATH"
            echo "   Download a Q4_K_M GGUF into models/gguf/ (see README) or set LLM_MODEL_PATH in .env"
            exit 1
        fi
    fi
    exec python server.py
fi

# ----------------------------- docker path (optional) -----------------------------
mkdir -p ./models/gguf

# Preflight for the default llama_cpp service: the GGUF is bind-mounted
# read-only from ./models/gguf, so it must exist before `compose up`.
BACKEND_DOCKER=${RUN_BACKEND:-llama_cpp}
if [ "$BACKEND_DOCKER" = "llama_cpp" ]; then
    MODEL_FILE=${LLM_MODEL_PATH:-models/gguf/Qwen3-1.7B-Q4_K_M.gguf}
    if [ ! -f "$MODEL_FILE" ]; then
        echo "❌ GGUF model not found: $MODEL_FILE"
        echo "   The llama_cpp container mounts ./models/gguf read-only (see README)"
        exit 1
    fi
fi

if [ "$FORCE_BUILD" = "1" ] || [ ! -f ".docker_built" ] || \
   find Dockerfile server.py backend app requirements.txt -newer .docker_built -print -quit 2>/dev/null | grep -q .; then
    echo "🔨 Building Docker image..."
    docker compose build
    touch .docker_built
else
    echo "✅ Using existing Docker image (use --force-build to rebuild)"
fi

if [ "$BACKEND_DOCKER" = "vllm" ]; then
    PROFILE="--profile vllm"
    TARGET_SERVICE="vllm"
    TARGET_CONTAINER="vllm_server"
    HEALTH_URL="http://localhost:${PORT}/v1/models"
elif [ "$BACKEND_DOCKER" = "transformers" ]; then
    PROFILE="--profile transformers"
    TARGET_SERVICE="transformers"
    TARGET_CONTAINER="transformers_server"
    HEALTH_URL="http://localhost:${PORT}/health"
else
    PROFILE=""
    TARGET_SERVICE="llama_cpp"
    TARGET_CONTAINER="llama_cpp_server"
    HEALTH_URL="http://localhost:${PORT}/health"
fi

echo "🚀 Starting container with backend: $BACKEND_DOCKER (host port $PORT)"
docker compose $PROFILE up -d

echo "⏳ Waiting for LLM server to become healthy..."
STATUS="starting"
for i in {1..20}; do
    STATUS=$(docker inspect --format='{{.State.Health.Status}}' $TARGET_CONTAINER 2>/dev/null || echo "starting")
    if [ "$STATUS" == "healthy" ]; then
        echo "✅ LLM server is healthy at: http://localhost:${PORT}"

        echo "🧪 Testing API endpoints..."
        if curl -s "$HEALTH_URL" > /dev/null; then
            echo "✅ Health/models endpoint working"
        fi

        echo "🔥 Sending warmup request..."
        WARMUP_ARGS=(-H "Content-Type: application/json"
          -d '{"model":"warmup","system":"Answer briefly.","prompt":"Say OK.","max_tokens":16}')
        if [ -n "${API_KEY:-}" ]; then
            WARMUP_ARGS+=(-H "Authorization: Bearer ${API_KEY}")
        fi
        curl -s "http://localhost:${PORT}/v1/completions" "${WARMUP_ARGS[@]}" > /dev/null \
          && echo "✅ Warmup successful" || echo "⚠️ Warmup failed"
        break
    else
        echo "   → Current status: $STATUS (retry $i/20)"
        sleep 10
    fi
done

if [ "$STATUS" != "healthy" ]; then
    echo "❌ Container failed to become healthy after 20 attempts"
    echo "📋 Container logs ($TARGET_CONTAINER):"
    docker compose logs "$TARGET_SERVICE" | tail -20
    exit 1
fi

AUTH_HEADER=""
if [ -n "${API_KEY:-}" ]; then
    # Literal $API_KEY in the hint below — it must not expand to the real key.
    AUTH_HEADER="-H 'Authorization: Bearer \$API_KEY' "
fi
echo "➡️ Example usage: curl http://localhost:${PORT}/v1/chat/completions ${AUTH_HEADER}-H 'Content-Type: application/json' -d '{\"model\":\"local\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":64}'"
