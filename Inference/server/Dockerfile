FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install Python + dependencies
RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Set Hugging Face token as environment variable
# (replace with your token or pass it at build/run time)
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
ENV HUGGINGFACE_HUB_CACHE="/app/hf_cache"

# Login once so it won’t ask again
RUN mkdir -p /root/.huggingface && \
    echo "{\"token\":\"${HF_TOKEN}\"}" > /root/.huggingface/token

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Expose API port
EXPOSE 1147

# Start server
CMD ["python3", "server.py", "--host", "0.0.0.0", "--port", "1147"]
