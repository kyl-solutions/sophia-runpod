# Sophia — ACE-Step 1.5 RunPod Serverless Image
#
# Cover mode on RunPod serverless (A100/A40 GPU).
# Cover mode only needs the DiT model (~4GB VRAM), no LLM required.
#
# ACE-Step 1.5 requires:
#   - Python 3.11 (strict)
#   - torch 2.10.0+cu128 from PyTorch CUDA 12.8 index
#   - CUDA 12.8 runtime

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3-pip \
        git \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# Install PyTorch 2.10.0 + CUDA 12.8 from PyTorch index FIRST
RUN pip install --no-cache-dir \
    torch==2.10.0+cu128 \
    torchvision==0.25.0+cu128 \
    torchaudio==2.10.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Install RunPod SDK + utilities
RUN pip install --no-cache-dir \
    runpod \
    soundfile \
    numpy \
    scipy

# Clone ACE-Step 1.5
RUN git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git /app/acestep
WORKDIR /app/acestep

# Install ACE-Step WITHOUT re-installing torch (already installed above)
# Use --no-deps first, then install remaining deps manually
RUN pip install --no-cache-dir --no-deps -e . && \
    pip install --no-cache-dir \
    accelerate>=1.12.0 \
    diffusers \
    diskcache \
    einops>=0.8.1 \
    fastapi>=0.110.0 \
    gradio>=6.5.1 \
    lightning>=2.0.0 \
    loguru>=0.7.3 \
    matplotlib>=3.7.5 \
    modelscope \
    numba>=0.63.1 \
    peft>=0.7.0 \
    toml \
    transformers>=4.51.0 \
    torchao \
    torchcodec \
    safetensors \
    huggingface_hub \
    uvicorn \
    vector_quantize_pytorch

# Install nano-vllm (bundled in ACE-Step repo)
RUN if [ -d "/app/acestep/nano_vllm" ]; then \
        pip install --no-cache-dir -e /app/acestep/nano_vllm; \
    fi

# Download model weights at build time (baked into image = faster cold-start)
ENV ACESTEP_DOWNLOAD_SOURCE=huggingface
WORKDIR /app/acestep
RUN python -c "\
from acestep.handler import AceStepHandler; \
h = AceStepHandler(); \
h.initialize_service(project_root='/app/acestep', config_path='acestep-v15-turbo', device='cpu'); \
print('Model downloaded successfully') \
" || echo "WARNING: Model download failed - will retry at runtime"

# Copy handler
WORKDIR /app
COPY handler.py /app/handler.py

# Environment
ENV ACESTEP_MODEL=acestep-v15-turbo
ENV ACESTEP_ROOT=/app/acestep
ENV PYTHONUNBUFFERED=1

# RunPod serverless entry point
CMD ["python", "/app/handler.py"]
