import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import requests
import torch
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, HttpUrl

MUSETALK_HOME = Path(os.getenv("MUSETALK_HOME", "/opt/MuseTalk")).resolve()
DATA_DIR = Path(os.getenv("DATA_DIR", "/data")).resolve()
VOICE_MODEL = Path(os.getenv("PIPER_VOICE", "/opt/voices/es_MX-ald-medium.onnx"))
API_TOKEN = os.getenv("API_TOKEN", "")
PORT = int(os.getenv("API_PORT", "8000"))
BATCH_SIZE = int(os.getenv("MUSETALK_BATCH_SIZE", "8"))
FPS = int(os.getenv("MUSETALK_FPS", "25"))

JOBS_DIR = DATA_DIR / "jobs"
SOURCES_DIR = DATA_DIR / "sources"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
SOURCES_DIR.mkdir(parents=True, exist_ok=True)

os.chdir(MUSETALK_HOME)
sys.path.insert(0, str(MUSETALK_HOME))

app = FastAPI(
    title="ARZ Video AI",
    version="0.1.0",
    description="Text-to-speech + MuseTalk 1.5 talking-avatar API.",
)

jobs = {}
jobs_lock = threading.Lock()
job_queue = queue.Queue()
avatar_cache = {}
engine_ready = False
rt = None


class GenerateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=3000)
    avatar_id: str = Field(default="default", min_length=1, max_length=40)
    avatar_url: Optional[HttpUrl] = None


def require_auth(authorization: Optional[str] = Header(default=None)):
    if not API_TOKEN:
        return
    expected = f"Bearer {API_TOKEN}"
    if authorization != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid bearer token",
        )


def clean_avatar_id(value: str) -> str:
    value = value.lower().strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", value):
        raise ValueError("avatar_id must use lowercase letters, numbers, _ or -")
    return value


def set_job(job_id: str, **changes):
    with jobs_lock:
        jobs.setdefault(job_id, {}).update(changes)


def get_job(job_id: str):
    with jobs_lock:
        item = jobs.get(job_id)
        return dict(item) if item else None


def init_engine():
    global rt, engine_ready

    import scripts.realtime_inference as realtime
    from musetalk.utils.audio_processor import AudioProcessor
    from musetalk.utils.face_parsing import FaceParsing
    from musetalk.utils.utils import load_all_model
    from transformers import WhisperModel

    rt = realtime
    rt.args = SimpleNamespace(
        version="v15",
        ffmpeg_path="/usr/bin",
        gpu_id=0,
        vae_type="sd-vae",
        unet_config=str(MUSETALK_HOME / "models/musetalkV15/musetalk.json"),
        unet_model_path=str(MUSETALK_HOME / "models/musetalkV15/unet.pth"),
        whisper_dir=str(MUSETALK_HOME / "models/whisper"),
        inference_config="",
        bbox_shift=0,
        result_dir="./results",
        extra_margin=10,
        fps=FPS,
        audio_padding_length_left=2,
        audio_padding_length_right=2,
        batch_size=BATCH_SIZE,
        output_vid_name=None,
        use_saved_coord=True,
        saved_coord=True,
        parsing_mode="jaw",
        left_cheek_width=90,
        right_cheek_width=90,
        skip_save_images=False,
    )

    rt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rt.vae, rt.unet, rt.pe = load_all_model(
        unet_model_path=rt.args.unet_model_path,
        vae_type=rt.args.vae_type,
        unet_config=rt.args.unet_config,
        device=rt.device,
    )
    rt.timesteps = torch.tensor([0], device=rt.device)

    # MuseTalk realtime mode is designed to use fp16 on NVIDIA GPUs.
    if torch.cuda.is_available():
        rt.pe = rt.pe.half().to(rt.device)
        rt.vae.vae = rt.vae.vae.half().to(rt.device)
        rt.unet.model = rt.unet.model.half().to(rt.device)
    else:
        rt.pe = rt.pe.to(rt.device)
        rt.vae.vae = rt.vae.vae.to(rt.device)
        rt.unet.model = rt.unet.model.to(rt.device)

    rt.audio_processor = AudioProcessor(feature_extractor_path=rt.args.whisper_dir)
    rt.weight_dtype = rt.unet.model.dtype
    rt.whisper = WhisperModel.from_pretrained(rt.args.whisper_dir)
    rt.whisper = rt.whisper.to(device=rt.device, dtype=rt.weight_dtype).eval()
    rt.whisper.requires_grad_(False)
    rt.fp = FaceParsing(
        left_cheek_width=rt.args.left_cheek_width,
        right_cheek_width=rt.args.right_cheek_width,
    )

    engine_ready = True


def avatar_base(avatar_id: str) -> Path:
    return MUSETALK_HOME / "results" / "v15" / "avatars" / avatar_id


def avatar_prepared(avatar_id: str) -> bool:
    base = avatar_base(avatar_id)
    return (
        (base / "latents.pt").exists()
        and (base / "coords.pkl").exists()
        and (base / "mask_coords.pkl").exists()
    )


def download_source(url: str, target: Path):
    with requests.get(url, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        total = 0
        with target.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > 250 * 1024 * 1024:
                    raise RuntimeError("avatar file is larger than 250 MB")
                f.write(chunk)


def normalize_avatar(source: Path, target: Path):
    # If OpenCV can decode the entire file as a still image, loop it into a short clip.
    is_image = cv2.imread(str(source), cv2.IMREAD_COLOR) is not None
    common = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
    ]
    vf = "fps=25,scale=trunc(iw/2)*2:trunc(ih/2)*2"
    if is_image:
        cmd = common + [
            "-loop", "1", "-i", str(source), "-t", "4",
            "-vf", vf, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(target),
        ]
    else:
        cmd = common + [
            "-i", str(source), "-t", "8",
            "-vf", vf, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(target),
        ]
    subprocess.run(cmd, check=True)


def ensure_avatar(avatar_id: str, avatar_url: Optional[str]):
    if avatar_id in avatar_cache:
        return avatar_cache[avatar_id]

    if avatar_prepared(avatar_id):
        avatar = rt.Avatar(
            avatar_id=avatar_id,
            video_path="",
            bbox_shift=0,
            batch_size=BATCH_SIZE,
            preparation=False,
        )
        avatar_cache[avatar_id] = avatar
        return avatar

    if not avatar_url:
        raise RuntimeError(
            f"avatar '{avatar_id}' is not prepared; provide avatar_url on the first job"
        )

    raw = SOURCES_DIR / f"{avatar_id}-{uuid.uuid4().hex}.source"
    normalized = SOURCES_DIR / f"{avatar_id}.mp4"
    download_source(avatar_url, raw)
    normalize_avatar(raw, normalized)
    raw.unlink(missing_ok=True)

    base = avatar_base(avatar_id)
    if base.exists():
        shutil.rmtree(base)

    avatar = rt.Avatar(
        avatar_id=avatar_id,
        video_path=str(normalized),
        bbox_shift=0,
        batch_size=BATCH_SIZE,
        preparation=True,
    )
    avatar_cache[avatar_id] = avatar
    return avatar


def synthesize_speech(text: str, output_wav: Path):
    cmd = [
        "piper",
        "--model", str(VOICE_MODEL),
        "--output_file", str(output_wav),
    ]
    subprocess.run(cmd, input=text, text=True, check=True)


def cleanup_outputs(keep: int = 30):
    files = sorted(JOBS_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[keep:]:
        old.unlink(missing_ok=True)


def run_one_job(job_id: str, payload: dict):
    set_job(job_id, status="running", started_at=time.time())
    wav_path = JOBS_DIR / f"{job_id}.wav"
    final_path = JOBS_DIR / f"{job_id}.mp4"

    try:
        synthesize_speech(payload["text"], wav_path)
        avatar = ensure_avatar(payload["avatar_id"], payload.get("avatar_url"))

        avatar.inference(
            audio_path=str(wav_path),
            out_vid_name=job_id,
            fps=FPS,
            skip_save_images=False,
        )

        produced = (
            avatar_base(payload["avatar_id"])
            / "vid_output"
            / f"{job_id}.mp4"
        )
        if not produced.exists():
            raise RuntimeError("MuseTalk did not create the expected MP4")

        shutil.copy2(produced, final_path)
        set_job(
            job_id,
            status="done",
            finished_at=time.time(),
            video_path=str(final_path),
            video_endpoint=f"/jobs/{job_id}/video",
        )
        cleanup_outputs()
    except Exception as exc:
        set_job(
            job_id,
            status="error",
            finished_at=time.time(),
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        wav_path.unlink(missing_ok=True)


def worker_loop():
    while True:
        job_id, payload = job_queue.get()
        try:
            run_one_job(job_id, payload)
        finally:
            job_queue.task_done()


@app.on_event("startup")
def startup():
    init_engine()
    threading.Thread(target=worker_loop, name="arz-video-worker", daemon=True).start()


@app.get("/")
def root():
    return {
        "service": "ARZ Video AI",
        "ready": engine_ready,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "ok": engine_ready,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "queued_jobs": job_queue.qsize(),
    }


@app.get("/avatars", dependencies=[Depends(require_auth)])
def avatars():
    root = MUSETALK_HOME / "results" / "v15" / "avatars"
    prepared = []
    if root.exists():
        for child in root.iterdir():
            if child.is_dir() and avatar_prepared(child.name):
                prepared.append(child.name)
    return {"avatars": sorted(prepared)}


@app.post("/jobs", status_code=202, dependencies=[Depends(require_auth)])
def create_job(request: GenerateRequest):
    try:
        avatar_id = clean_avatar_id(request.avatar_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    avatar_url = str(request.avatar_url) if request.avatar_url else None
    if not avatar_prepared(avatar_id) and avatar_id not in avatar_cache and not avatar_url:
        raise HTTPException(
            status_code=400,
            detail="Provide avatar_url the first time this avatar_id is used.",
        )

    job_id = uuid.uuid4().hex
    payload = {
        "text": request.text.strip(),
        "avatar_id": avatar_id,
        "avatar_url": avatar_url,
    }
    set_job(
        job_id,
        id=job_id,
        status="queued",
        created_at=time.time(),
        avatar_id=avatar_id,
    )
    job_queue.put((job_id, payload))
    return {
        "id": job_id,
        "status": "queued",
        "status_endpoint": f"/jobs/{job_id}",
        "video_endpoint": f"/jobs/{job_id}/video",
    }


@app.get("/jobs/{job_id}", dependencies=[Depends(require_auth)])
def job_status(job_id: str):
    item = get_job(job_id)
    if not item:
        raise HTTPException(status_code=404, detail="Job not found")
    item.pop("video_path", None)
    return item


@app.get("/jobs/{job_id}/video", dependencies=[Depends(require_auth)])
def job_video(job_id: str):
    item = get_job(job_id)
    if not item:
        raise HTTPException(status_code=404, detail="Job not found")
    if item.get("status") != "done":
        raise HTTPException(status_code=409, detail=f"Job status is {item.get('status')}")
    path = Path(item["video_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        app_dir="/app",
        host="::",
        port=PORT,
        workers=1,
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
