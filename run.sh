#!/bin/bash
set -e

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
        
        # Export the variable if it contains an equals sign
        if [[ "$line" =~ ^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=[[:space:]]* ]]; then
            export "$line"
        fi
    done < .env
    
    # Debug: Show loaded variables
    echo "📋 Loaded environment variables:"
    echo "   HF_TOKEN: ${HF_TOKEN:0:10}..."
    echo "   VLLM_MODELS: $VLLM_MODELS"
    echo "   VLLM_QUANTIZATION: $VLLM_QUANTIZATION"
    echo "   VLLM_LOAD_ON_START: $VLLM_LOAD_ON_START"
    echo "   VLLM_PORT: $VLLM_PORT"
fi

# Now that env vars are loaded, print models
echo "🚀 Starting vLLM server with models: $VLLM_MODELS"

# Ensure models cache dir exists
mkdir -p ./models

# Check if we need to build
if [ "$1" = "--force-build" ] || [ "$1" = "-f" ]; then
    echo "🔨 Force building custom Docker image..."
    docker compose build --no-cache
elif [ ! -f ".docker_built" ] || [ "Dockerfile" -nt ".docker_built" ] || [ "server.py" -nt ".docker_built" ]; then
    echo "🔨 Building custom Docker image (Dockerfile or server.py changed)..."
    docker compose build
    touch .docker_built
else
    echo "✅ Using existing Docker image (use --force-build to rebuild)"
fi

BACKEND=${RUN_BACKEND:-transformers}
if [ "$BACKEND" = "vllm" ]; then
  PROFILE="--profile vllm"
  HEALTH_URL="http://localhost:${VLLM_PORT}/v1/models"
else
  PROFILE="--profile transformers"
  HEALTH_URL="http://localhost:${VLLM_PORT}/health"
fi

echo "🚀 Starting container with backend: $BACKEND"
docker compose $PROFILE up -d

# Wait for healthcheck
echo "⏳ Waiting for LLM server to become healthy..."
for i in {1..20}; do
    TARGET_CONTAINER=$([ "$BACKEND" = "vllm" ] && echo vllm_server || echo transformers_server)
    STATUS=$(docker inspect --format='{{.State.Health.Status}}' $TARGET_CONTAINER 2>/dev/null || echo "starting")
    if [ "$STATUS" == "healthy" ]; then
        echo "✅ LLM server is healthy at: http://localhost:${VLLM_PORT}"
        
        # Test the API
        echo "🧪 Testing API endpoints..."
        if curl -s $HEALTH_URL > /dev/null; then
            echo "✅ Models endpoint working"
        fi
        
        # Warmup request
        echo "🔥 Sending warmup request..."
        curl -s http://localhost:${VLLM_PORT}/v1/completions \
          -H "Content-Type: application/json" \
          -d "{\"model\":\"google/flan-t5-small\",\"system\":\"Answer briefly.\",\"prompt\":\"warmup\",\"max_tokens\":16}" > /dev/null && echo "✅ Warmup successful" || echo "⚠️ Warmup failed"
        break
    else
        echo "   → Current status: $STATUS (retry $i/20)"
        sleep 10
    fi
done

if [ "$STATUS" != "healthy" ]; then
    echo "❌ Container failed to become healthy after 20 attempts"
    echo "📋 Container logs:"
    docker compose logs vllm | tail -20
    exit 1
fi

echo "➡️ Example usage: curl http://localhost:${VLLM_PORT}/v1/completions -H 'Content-Type: application/json' -d '{\"model\":\"google/flan-t5-small\",\"system\":\"Answer briefly.\",\"prompt\":\"Hello, how are you?\",\"max_tokens\":64}'"
