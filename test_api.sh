#!/bin/bash

# OpenAI API Compatible Local LLM Server Test Script
# Replace localhost:8000 with your server address if different

BASE_URL="http://localhost:8000"

echo "🚀 Testing OpenAI Compatible Local LLM Server"
echo "=============================================="

# Test 1: Basic Health Check
echo "1. Health Check"
curl -s "$BASE_URL/health" && echo -e "\n✅ Health check passed\n" || echo "❌ Health check failed\n"

# Test 2: List Models
echo "2. List Available Models"
curl -s "$BASE_URL/v1/models" | jq '.' && echo -e "\n✅ Models endpoint working\n" || echo "❌ Models endpoint failed\n"

# Test 3: Basic Text Completion
echo "3. Basic Text Completion"
curl -s "$BASE_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "prompt": "The future of artificial intelligence is",
    "max_tokens": 50,
    "temperature": 0.7
  }' | jq '.' && echo -e "\n✅ Text completion working\n" || echo "❌ Text completion failed\n"

# Test 4: Text Completion with Advanced Parameters
echo "4. Text Completion with Advanced Parameters"
curl -s "$BASE_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "prompt": "Once upon a time in a distant galaxy",
    "max_tokens": 100,
    "temperature": 0.8,
    "top_p": 0.9,
    "stop": ["\n", "The end"]
  }' | jq '.' && echo -e "\n✅ Advanced text completion working\n" || echo "❌ Advanced text completion failed\n"

# Test 5: Chat Completion with System Prompt
echo "5. Chat Completion with System Prompt"
curl -s "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant. Provide clear and concise answers."},
      {"role": "user", "content": "How do I create a Python function?"}
    ],
    "max_tokens": 150,
    "temperature": 0.5
  }' | jq '.' && echo -e "\n✅ Chat completion with system prompt working\n" || echo "❌ Chat completion failed\n"

# Test 6: Multi-turn Conversation
echo "6. Multi-turn Conversation"
curl -s "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "messages": [
      {"role": "system", "content": "You are a friendly assistant."},
      {"role": "user", "content": "Hello! How are you today?"},
      {"role": "assistant", "content": "Hello! I'\''m doing well, thank you for asking. How can I help you today?"},
      {"role": "user", "content": "Can you tell me a short joke?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }' | jq '.' && echo -e "\n✅ Multi-turn conversation working\n" || echo "❌ Multi-turn conversation failed\n"

# Test 7: Creative Writing with High Temperature
echo "7. Creative Writing (High Temperature)"
curl -s "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "messages": [
      {"role": "system", "content": "You are a creative writer."},
      {"role": "user", "content": "Write a short story about a robot learning to paint."}
    ],
    "max_tokens": 200,
    "temperature": 1.2,
    "top_p": 0.95
  }' | jq '.' && echo -e "\n✅ Creative writing working\n" || echo "❌ Creative writing failed\n"

# Test 8: Code Generation
echo "8. Code Generation"
curl -s "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "messages": [
      {"role": "system", "content": "You are a Python programming expert."},
      {"role": "user", "content": "Write a Python function to calculate fibonacci numbers."}
    ],
    "max_tokens": 150,
    "temperature": 0.3
  }' | jq '.' && echo -e "\n✅ Code generation working\n" || echo "❌ Code generation failed\n"

# Test 9: Structured Output Request
echo "9. Structured Output Request"
curl -s "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant that provides structured information."},
      {"role": "user", "content": "List 3 benefits of renewable energy."}
    ],
    "max_tokens": 100,
    "temperature": 0.5
  }' | jq '.' && echo -e "\n✅ Structured output working\n" || echo "❌ Structured output failed\n"

# Test 10: Error Handling Test
echo "10. Error Handling Test (Invalid Model)"
curl -s "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "invalid-model",
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }' | jq '.' && echo -e "\n✅ Error handling working\n" || echo "❌ Error handling failed\n"

echo "🎉 All tests completed!"
echo "Your local LLM server is OpenAI API compatible! 🚀"
