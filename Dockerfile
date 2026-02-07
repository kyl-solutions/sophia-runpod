# Sophia — ACE-Step 1.5 RunPod Serverless Image
#
# Cover mode on RunPod serverless (A100/A40 GPU).
# Cover mode only needs the DiT model (~4GB VRAM), no LLM required.

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    soundfile \
    numpy \
    scipy

# Clone ACE-Step 1.5 and install
RUN git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git /app/acestep
WORKDIR /app/acestep
RUN pip install --no-cache-dir -e .

# Download model weights at build time (baked into image = zero cold-start)
ENV ACESTEP_DOWNLOAD_SOURCE=huggingface
RUN python -c "\
from acestep.handler import AceStepHandler; \
h = AceStepHandler(); \
h.initialize_service(project_root='./', config_path='acestep-v15-turbo', device='cpu'); \
print('Model downloaded successfully') \
" || echo "Model download will happen on first run"

# Copy handler
WORKDIR /app
COPY handler.py /app/handler.py

# Environment
ENV ACESTEP_MODEL=acestep-v15-turbo
ENV PYTHONUNBUFFERED=1

# RunPod serverless entry point
CMD ["python", "/app/handler.py"]
