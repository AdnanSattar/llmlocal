# Local LLM API — CPU (Transformers) & GPU (vLLM)

OpenAI-compatible local inference with Docker. **Default: CPU (Transformers).** Set `RUN_BACKEND=vllm` when a GPU is available.

This project provides **OpenAI-compatible `/v1` endpoints** for self-hosted LLMs. Use the lightweight [Transformers](https://huggingface.co/docs/transformers) backend on CPU today, or switch to [vLLM](https://github.com/vllm-project/vllm) via `RUN_BACKEND` when you have suitable GPU hardware. Models such as [google/flan-t5-small](https://huggingface.co/google/flan-t5-small) (CPU) or Qwen/Llama (GPU) are configurable through `.env` and exposed through a single API surface.

---

## 🔧 What’s Included

- **Docker Compose setup** → runs `vLLM` server inside a container.  
- **Configurable via `.env`** → choose models, quantization level (int4/int8), preload option, and port.  
- **Model caching** → models are stored in `./models` so they don’t re-download every time.  
- **Warmup request** → prevents first-request lag if preload is disabled.  
- **Healthcheck** → waits for the container to be healthy before first use.  

---

## 📂 Files

- `docker-compose.yml` → runs vLLM server with your chosen models.  
- `.env` → holds configs (models, quantization, preload flag, port).  
- `run.sh` → startup script, handles healthcheck + warmup request.  

---

## ▶️ How to Run

1. **Clone repo & enter folder**  

   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   ```

2. **Configure `.env`**  

   Minimal variables (examples):

   ```bash

   # Backend: transformers (CPU, default) | vllm (GPU-ready)
   RUN_BACKEND=transformers
   VLLM_PORT=8000
   # Optional: HF token for private/model downloads
   HF_TOKEN=
   # For vLLM (when GPU available)
   VLLM_MODEL=Qwen/Qwen3-0.6B
   ```

3. **Start (CPU, Transformers backend)**  

   ```bash

   chmod +x run.sh
   ./run.sh
   # or explicitly
   RUN_BACKEND=transformers ./run.sh
   ```

4. **Start (GPU, vLLM backend when hardware is ready)**  

   ```bash
   RUN_BACKEND=vllm VLLM_MODEL=Qwen/Qwen3-0.6B ./run.sh
   ```

   vLLM Docker usage reference:
   <https://docs.vllm.ai/en/stable/deployment/docker.html#use-the-custom-built-vllm-docker-image>

5. **Health checks**  

   ```bash
   # Transformers
   curl http://localhost:8000/health
   # vLLM
   curl http://localhost:8000/v1/models
   ```

6. **Test completions (OpenAI-style)**  

   ```bash

   curl http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "google/flan-t5-small",
       "system": "Answer briefly.",
       "prompt": "Hello, how are you?",
       "max_tokens": 64
     }'
   ```

---

## ✅ What You Should Do Next

- Make sure you **have a Hugging Face token** in `.env` so models can be downloaded.  
- Test both models (`qwen-1.5b` and `llama-1b`) to confirm they’re responding.  
- Adjust `.env` to try different quantization (`int8`, `int4`) depending on GPU memory.  

---

## ⚙️ Alternative: Minimal Transformers-based server (/v1/completions)

If GPU constraints or vLLM incompatibilities block progress, you can run a lightweight, CPU‑friendly server using HuggingFace Transformers. This provides an OpenAI‑style `/v1/completions` endpoint without vLLM.

- Image: custom Dockerfile (PyTorch + Transformers)
- Endpoint: `/v1/completions` (OpenAI compatible)
- Default model: `google/flan-t5-small` (instruction‑tuned; good on CPU)

### How to Run

1 Build and start (same compose stack):

```bash
docker compose build
docker compose up -d
```

2 Health and models:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

3 Completion example:

```bash

curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/flan-t5-small",
    "system": "Answer clearly and briefly.",
    "prompt": "What is an API?",
    "max_tokens": 64,
    "temperature": 0.3
  }'
```

Notes:

- This path is ideal for PoC and CPU‑only environments. For production throughput and larger models, vLLM on a modern GPU is recommended.
