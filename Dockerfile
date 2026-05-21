FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Install transformers and dependencies
RUN pip install transformers

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy server script
COPY server.py /app/server.py

# Expose port
EXPOSE 8000

# Run the server
CMD ["python", "/app/server.py"]
