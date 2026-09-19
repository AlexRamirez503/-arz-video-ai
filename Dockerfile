FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MUSETALK_HOME=/opt/MuseTalk \
    DATA_DIR=/data \
    API_PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git curl wget ca-certificates ffmpeg \
    build-essential ninja-build pkg-config \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libsndfile1 espeak-ng \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python && \
    python -m pip install --upgrade pip setuptools wheel

# MuseTalk recommends PyTorch 2.0.1 with CUDA 11.8 wheels.
RUN python -m pip install \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

ARG MUSETALK_COMMIT=0a89dec45a0192b824e3cf4daf96c239440c5ed8
RUN git clone https://github.com/TMElyralab/MuseTalk.git /opt/MuseTalk && \
    cd /opt/MuseTalk && \
    git checkout "${MUSETALK_COMMIT}"

WORKDIR /opt/MuseTalk

# Install the official inference dependencies, omitting training/demo-only packages.
RUN grep -v -E '^(tensorflow|tensorboard|gradio)' requirements.txt > /tmp/musetalk-inference.txt && \
    python -m pip install -r /tmp/musetalk-inference.txt && \
    python -m pip install -U openmim && \
    mim install mmengine && \
    mim install "mmcv==2.0.1" && \
    mim install "mmdet==3.1.0" && \
    mim install "mmpose==1.1.0"

# API + local Spanish TTS.
COPY requirements-api.txt /tmp/requirements-api.txt
RUN python -m pip install -r /tmp/requirements-api.txt

# Download only the weights used by MuseTalk 1.5 inference.
RUN mkdir -p models/musetalkV15 models/sd-vae models/whisper models/dwpose models/face-parse-bisent && \
    python -m pip install -U "huggingface_hub[hf_xet]" && \
    huggingface-cli download TMElyralab/MuseTalk \
      --local-dir models \
      --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth" && \
    huggingface-cli download stabilityai/sd-vae-ft-mse \
      --local-dir models/sd-vae \
      --include "config.json" "diffusion_pytorch_model.bin" && \
    huggingface-cli download openai/whisper-tiny \
      --local-dir models/whisper \
      --include "config.json" "pytorch_model.bin" "preprocessor_config.json" && \
    huggingface-cli download yzd-v/DWPose \
      --local-dir models/dwpose \
      --include "dw-ll_ucoco_384.pth" && \
    huggingface-cli download ManyOtherFunctions/face-parse-bisent \
      --local-dir models/face-parse-bisent \
      --include "79999_iter.pth" "resnet18-5c106cde.pth"

# Piper Mexican Spanish voice.
RUN mkdir -p /opt/voices && \
    wget -q -O /opt/voices/es_MX-ald-medium.onnx \
      https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/ald/medium/es_MX-ald-medium.onnx && \
    wget -q -O /opt/voices/es_MX-ald-medium.onnx.json \
      https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/ald/medium/es_MX-ald-medium.onnx.json

# Put generated/cached files outside the application tree.
RUN mkdir -p /data/results /data/jobs /data/sources && \
    rm -rf /opt/MuseTalk/results && \
    ln -s /data/results /opt/MuseTalk/results

COPY server /app
ENV PYTHONPATH=/opt/MuseTalk

EXPOSE 8000
WORKDIR /opt/MuseTalk

CMD ["python", "/app/main.py"]
