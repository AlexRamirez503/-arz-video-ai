FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MUSETALK_HOME=/opt/MuseTalk \
    DATA_DIR=/data \
    API_PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-venv \
    git curl wget ca-certificates ffmpeg \
    build-essential ninja-build pkg-config \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libsndfile1 espeak-ng fonts-dejavu-core \
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
    python -m pip install "setuptools<82" wheel && \
    python -m pip install --no-build-isolation "chumpy==0.70" && \
    mim install "mmpose==1.1.0"

# API + local Spanish TTS.
COPY requirements-api.txt /tmp/requirements-api.txt
RUN python -m pip install -r /tmp/requirements-api.txt

# Fast chat mode deliberately omits Wan2.1 and MimicMotion. Both require large
# additional model stacks and make interactive jobs impractical on a 12 GB GPU.
# The image ships a reusable animated 3D source clip and relies on MuseTalk.

# Download only the weights used by MuseTalk 1.5 inference.
RUN mkdir -p models/musetalkV15 models/sd-vae models/whisper models/dwpose models/face-parse-bisent && \
    python -m pip install "huggingface_hub==0.30.2" && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('TMElyralab/MuseTalk', local_dir='models', allow_patterns=['musetalkV15/musetalk.json','musetalkV15/unet.pth'])" && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('stabilityai/sd-vae-ft-mse', local_dir='models/sd-vae', allow_patterns=['config.json','diffusion_pytorch_model.bin'])" && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/whisper-tiny', local_dir='models/whisper', allow_patterns=['config.json','pytorch_model.bin','preprocessor_config.json'])" && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('yzd-v/DWPose', local_dir='models/dwpose', allow_patterns=['dw-ll_ucoco_384.pth'])" && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('ManyOtherFunctions/face-parse-bisent', local_dir='models/face-parse-bisent', allow_patterns=['79999_iter.pth','resnet18-5c106cde.pth'])"

# Piper Spanish voices. The Studio defaults to the masculine Davefx voice while
# retaining the Mexican Ald voice as an optional feminine choice.
RUN mkdir -p /opt/voices && \
    wget -q -O /opt/voices/es_MX-ald-medium.onnx \
      https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/ald/medium/es_MX-ald-medium.onnx && \
    wget -q -O /opt/voices/es_MX-ald-medium.onnx.json \
      https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/ald/medium/es_MX-ald-medium.onnx.json && \
    wget -q -O /opt/voices/es_ES-davefx-medium.onnx \
      https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx && \
    wget -q -O /opt/voices/es_ES-davefx-medium.onnx.json \
      https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json

# Put generated/cached files outside the application tree.
RUN mkdir -p /data/results /data/jobs /data/sources /data/models && \
    rm -rf /opt/MuseTalk/results && \
    ln -s /data/results /opt/MuseTalk/results

COPY server /app
RUN python -m py_compile /app/main.py /app/motion.py /app/motion_inference.py
ENV PYTHONPATH=/opt/MuseTalk

EXPOSE 8000
WORKDIR /opt/MuseTalk

CMD ["python", "/app/main.py"]
# Build trigger
