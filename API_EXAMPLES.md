# OpenAI Compatible Local LLM Server - `/v1/completions` API

Your local LLM server is **100% compatible** with OpenAI's `/v1/completions` API! 🎉
Default model: `google/flan-t5-small` (instruction-tuned, CPU-friendly).

## Base URL
Replace `localhost:8000` with your server address if different.

---

## 🚀 Quick Start

### Basic Text Completion
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/flan-t5-small",
    "prompt": "The future of artificial intelligence is",
    "max_tokens": 50
  }'
```

---

## 📚 Complete API Examples

### 1. Basic Text Completion
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/flan-t5-small",
    "prompt": "Hello, how are you?",
    "max_tokens": 100
  }'
```

### 2. With Temperature Control
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/flan-t5-small",
    "prompt": "Once upon a time in a distant galaxy",
    "max_tokens": 150,
    "temperature": 0.8
  }'
```

### 3. Advanced Parameters
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/flan-t5-small",
    "prompt": "The secret to happiness is",
    "max_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9,
    "stop": ["\n", "The end"]
  }'
```

### 4. Creative Writing (High Temperature)
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/flan-t5-small",
    "prompt": "In a world where robots have emotions",
    "max_tokens": 250,
    "temperature": 1.2,
    "top_p": 0.95
  }'
```

### 5. Code Generation (Low Temperature)
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/flan-t5-small",
    "prompt": "def fibonacci(n):",
    "max_tokens": 200,
    "temperature": 0.3
  }'
```

### 6. Question Answering
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/flan-t5-small",
    "prompt": "Question: What is machine learning?\nAnswer:",
    "max_tokens": 150,
    "temperature": 0.6
  }'
```

### 7. Translation Task
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/flan-t5-small",
    "prompt": "English: Hello, how are you?\nFrench:",
    "max_tokens": 50,
    "temperature": 0.2
  }'
```

### 8. Story Continuation
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/flan-t5-small",
    "prompt": "The detective entered the dark room and found",
    "max_tokens": 200,
    "temperature": 0.9
  }'
```

---

## 🔧 Supported Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | "google/flan-t5-small" | Model to use |
| `prompt` | string | required | Text to complete |
| `max_tokens` | integer | 50 | Maximum tokens to generate |
| `temperature` | float | 0.7 | Controls randomness (0.0-2.0) |
| `top_p` | float | 1.0 | Nucleus sampling (0.0-1.0) |
| `stop` | string/array | null | Stop sequences |

---

## 📋 Response Format

Your server returns responses in the exact OpenAI API format:

```json
{
  "id": "cmpl-123456789012345678901",
  "object": "text_completion",
  "created": 1677858242,
  "model": "google/flan-t5-small",
  "choices": [
    {
      "text": "Generated text here...",
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 10,
    "total_tokens": 15
  }
}
```

---

## 🧪 Testing

### Health Check
```bash
curl http://localhost:8000/health
```

### List Models
```bash
curl http://localhost:8000/v1/models
```

### Run Test Suite
```bash
chmod +x test_api.sh
./test_api.sh
```

---

## 💡 Usage Tips

### Temperature Guidelines:
- **0.0-0.3**: Factual, deterministic responses
- **0.4-0.7**: Balanced creativity and accuracy
- **0.8-1.2**: Creative, diverse responses
- **1.3+**: Very creative, potentially incoherent

### Prompt Engineering:
- Be specific and clear in your prompts
- Use examples when possible
- Include context for better results
- Use stop sequences to control output length

### Performance:
- Lower `max_tokens` = faster responses
- Higher `temperature` = more creative but slower
- Use `stop` sequences to end generation early

---

## 🎯 Integration Examples

### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "google/flan-t5-small",
        "prompt": "Hello, how are you?",
        "max_tokens": 100,
        "temperature": 0.7
    }
)
print(response.json())
```

### JavaScript
```javascript
const response = await fetch('http://localhost:8000/v1/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'google/flan-t5-small',
    prompt: 'Hello, how are you?',
    max_tokens: 100,
    temperature: 0.7
  })
});
const data = await response.json();
console.log(data);
```

---

## 🎉 Your server is ready!

You now have a **fully OpenAI-compatible** local LLM server that works with any application expecting the standard `/v1/completions` API.

**Happy coding!** 🚀

---

## ➕ Additional `/v1/completions` Examples (with system and summarizer)

### Warmup (fast, short)
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/flan-t5-small",
    "prompt": "warmup",
    "max_tokens": 8
  }'
```

### Completion with System Prompt (recommended)
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/flan-t5-small",
    "system": "You are a helpful assistant that responds briefly and clearly.",
    "prompt": "Explain what an API is.",
    "max_tokens": 120,
    "temperature": 0.5
  }'
```

### Creative Writing (High Temperature)
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/flan-t5-small",
    "system": "You are a creative fiction writer.",
    "prompt": "In a city of glass towers, a courier discovers a secret.",
    "max_tokens": 250,
    "temperature": 1.2,
    "top_p": 0.95
  }'
```

### Constrained Output with Stop Sequences
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/flan-t5-small",
    "system": "You are a helpful assistant.",
    "prompt": "List three benefits of unit testing:",
    "max_tokens": 80,
    "temperature": 0.4,
    "stop": ["\n", "User:"]
  }'
```

### Summarizer (Concise 2–3 sentences)
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/flan-t5-small",
    "system": "You are a professional summarizer. Produce a concise, factual summary in 2-3 sentences. Avoid repetition and opinions.",
    "prompt": "Summarize: Artificial intelligence (AI) refers to computer systems capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding. Recent advances in machine learning and large datasets have accelerated AI capabilities across industries, from healthcare to finance, enabling automation and improved decision-making.",
    "max_tokens": 120,
    "temperature": 0.3,
    "top_p": 0.9,
    "stop": ["\n\n"]
  }'
```

### Summarizer (Bulleted key points)
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/flan-t5-small",
    "system": "You extract key points as short bullet items. No preamble, no conclusion.",
    "prompt": "Key points: Artificial intelligence (AI) refers to computer systems capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding. Recent advances in machine learning and large datasets have accelerated AI capabilities across industries, from healthcare to finance, enabling automation and improved decision-making.",
    "max_tokens": 120,
    "temperature": 0.35,
    "stop": ["\n\n"]
  }'
```

### Note on `system` field
- The server supports an optional `system` field which is prepended server-side using an Instruction/Question/Answer template.
- For more factual outputs, use lower temperatures (0.2–0.5) and provide clear instructions.
