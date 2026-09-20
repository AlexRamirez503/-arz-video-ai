import gc
import hmac
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
from typing import Literal, Optional

import cv2
import requests
import torch
from fastapi import Cookie, Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field, HttpUrl

from motion import motion_available, run_motion, save_upload

MUSETALK_HOME = Path(os.getenv("MUSETALK_HOME", "/opt/MuseTalk")).resolve()
DATA_DIR = Path(os.getenv("DATA_DIR", "/data")).resolve()
VOICE_MODEL = Path(os.getenv("PIPER_VOICE", "/opt/voices/es_MX-ald-medium.onnx"))
API_TOKEN = os.getenv("API_TOKEN", "")
STUDIO_ACCESS_KEY = os.getenv("STUDIO_ACCESS_KEY", "")
STUDIO_SESSION_COOKIE = "aztv_studio_session"
STUDIO_SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 30
PORT = int(os.getenv("API_PORT", "8000"))
BATCH_SIZE = int(os.getenv("MUSETALK_BATCH_SIZE", "8"))
FPS = int(os.getenv("MUSETALK_FPS", "25"))
WAN_HOME = Path(os.getenv("WAN_HOME", "/opt/Wan2.1")).resolve()
WAN_PYTHON = Path(os.getenv("WAN_PYTHON", "/opt/wan-venv/bin/python")).resolve()
WAN_MODEL_DIR = Path(
    os.getenv("WAN_MODEL_DIR", "/data/models/Wan2.1-T2V-1.3B")
).resolve()
WAN_MODEL_ID = os.getenv("WAN_MODEL_ID", "Wan-AI/Wan2.1-T2V-1.3B")
WAN_MODEL_MARKER = WAN_MODEL_DIR / ".download-complete"
WAN_MODEL_MIN_FREE_GB = int(os.getenv("WAN_MODEL_MIN_FREE_GB", "12"))
FAST_PROMO_AVATAR_ID = os.getenv("FAST_PROMO_AVATAR_ID", "promo3d")
FAST_PROMO_AVATAR_PATH = Path(
    os.getenv("FAST_PROMO_AVATAR_PATH", "/app/assets/default_3d_presenter.mp4")
).resolve()
FAST_VOICE_LENGTH_SCALE = float(os.getenv("FAST_VOICE_LENGTH_SCALE", "0.84"))
NORMAL_VOICE_LENGTH_SCALE = float(os.getenv("NORMAL_VOICE_LENGTH_SCALE", "1.0"))

JOBS_DIR = DATA_DIR / "jobs"
SOURCES_DIR = DATA_DIR / "sources"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
SOURCES_DIR.mkdir(parents=True, exist_ok=True)

os.chdir(MUSETALK_HOME)
sys.path.insert(0, str(MUSETALK_HOME))

app = FastAPI(
    title="ARZ Video AI",
    version="0.1.0",
    description="Fast animated-avatar + local Spanish voice API.",
)

jobs = {}
jobs_lock = threading.Lock()
job_queue = queue.Queue()
avatar_cache = {}
wan_lock = threading.Lock()
engine_ready = False
rt = None


class GenerateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=3000)
    avatar_id: str = Field(default="default", min_length=1, max_length=40)
    avatar_url: Optional[HttpUrl] = None


class PromoRequest(BaseModel):
    text: str = Field(min_length=1, max_length=800)
    person_prompt: str = Field(
        default=(
            "a friendly young adult presenter in modern casual clothing, "
            "looking directly at the camera"
        ),
        min_length=3,
        max_length=800,
    )
    brand_text: str = Field(default="AZTV", min_length=1, max_length=80)
    cta_text: str = Field(default="Descárgala hoy", max_length=120)
    logo_url: Optional[HttpUrl] = None
    orientation: Literal["vertical", "landscape"] = "vertical"
    steps: int = Field(default=28, ge=20, le=50)
    seed: int = Field(default=-1, ge=-1)


class StudioRequest(BaseModel):
    message: str = Field(min_length=3, max_length=1800)
    avatar_url: Optional[HttpUrl] = None
    brand_text: str = Field(default="AZTV", min_length=1, max_length=80)
    cta_text: str = Field(default="Descárgala hoy", max_length=120)
    voice_speed: Literal["fast", "normal"] = "fast"


def has_studio_session(studio_session: Optional[str]) -> bool:
    return bool(
        STUDIO_ACCESS_KEY
        and studio_session
        and hmac.compare_digest(studio_session, STUDIO_ACCESS_KEY)
    )


def require_auth(
    authorization: Optional[str] = Header(default=None),
    studio_session: Optional[str] = Cookie(default=None, alias=STUDIO_SESSION_COOKIE),
):
    if has_studio_session(studio_session):
        return
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

    if engine_ready and rt is not None:
        return

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


def release_musetalk_engine():
    global rt, engine_ready

    avatar_cache.clear()
    if rt is not None:
        for name in ("vae", "unet", "pe", "whisper", "fp", "audio_processor"):
            if hasattr(rt, name):
                setattr(rt, name, None)
    rt = None
    engine_ready = False
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


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


def synthesize_speech(
    text: str,
    output_wav: Path,
    length_scale: float = NORMAL_VOICE_LENGTH_SCALE,
):
    cmd = [
        "piper",
        "--model", str(VOICE_MODEL),
        "--output_file", str(output_wav),
        "--length-scale", str(length_scale),
        "--sentence-silence", "0.08",
    ]
    subprocess.run(cmd, input=text, text=True, check=True)


def wan_model_ready() -> bool:
    return WAN_MODEL_MARKER.exists()


def ensure_wan_model(job_id: str):
    if wan_model_ready():
        return

    WAN_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(WAN_MODEL_DIR.parent).free
    required = WAN_MODEL_MIN_FREE_GB * 1024 * 1024 * 1024
    if free_bytes < required:
        free_gb = free_bytes / (1024 ** 3)
        raise RuntimeError(
            f"Wan2.1 needs at least {WAN_MODEL_MIN_FREE_GB} GB free before the "
            f"first model download; only {free_gb:.1f} GB is available"
        )

    set_job(job_id, stage="downloading_video_model")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=WAN_MODEL_ID,
        local_dir=str(WAN_MODEL_DIR),
        max_workers=4,
    )
    WAN_MODEL_MARKER.write_text(WAN_MODEL_ID + "\n")


def build_person_prompt(person_prompt: str) -> str:
    return (
        "Photorealistic commercial social-media advertisement video. "
        f"One {person_prompt}. "
        "The presenter faces the camera and speaks naturally with small expressive "
        "hand gestures, natural head movement, blinking and subtle body movement. "
        "Medium shot, stable camera, one continuous shot, clean modern background, "
        "realistic skin and hands, professional advertising lighting. "
        "No subtitles, no logos, no text, no scene cuts, no extra people."
    )


def run_wan_video(
    job_id: str,
    person_prompt: str,
    orientation: str,
    steps: int,
    seed: int,
) -> Path:
    with wan_lock:
        release_musetalk_engine()
        ensure_wan_model(job_id)
        set_job(job_id, stage="generating_person_video")

        size = "480*832" if orientation == "vertical" else "832*480"
        output = SOURCES_DIR / f"{job_id}-wan.mp4"
        log_path = JOBS_DIR / f"{job_id}-wan.log"
        prompt = build_person_prompt(person_prompt)

        cmd = [
            str(WAN_PYTHON),
            str(WAN_HOME / "generate.py"),
            "--task", "t2v-1.3B",
            "--size", size,
            "--ckpt_dir", str(WAN_MODEL_DIR),
            "--offload_model", "True",
            "--t5_cpu",
            "--sample_shift", "8",
            "--sample_guide_scale", "6",
            "--sample_steps", str(steps),
            "--frame_num", "81",
            "--prompt", prompt,
            "--save_file", str(output),
        ]
        if seed >= 0:
            cmd.extend(["--base_seed", str(seed)])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["OMP_NUM_THREADS"] = "4"

        with log_path.open("w") as log:
            result = subprocess.run(
                cmd,
                cwd=str(WAN_HOME),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=5400,
            )

        if result.returncode != 0 or not output.exists():
            tail = ""
            try:
                tail = log_path.read_text(errors="replace")[-3000:]
            except Exception:
                pass
            raise RuntimeError(
                "Wan2.1 video generation failed. Last log output: " + tail
            )
        return output


def prepare_avatar_from_local(avatar_id: str, source: Path):
    init_engine()
    normalized = SOURCES_DIR / f"{avatar_id}.mp4"
    normalize_avatar(source, normalized)

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


def _download_logo(url: str, target: Path):
    with requests.get(url, stream=True, timeout=(15, 60)) as response:
        response.raise_for_status()
        total = 0
        with target.open("wb") as f:
            for chunk in response.iter_content(chunk_size=256 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > 20 * 1024 * 1024:
                    raise RuntimeError("logo file is larger than 20 MB")
                f.write(chunk)


def apply_branding(
    source: Path,
    output: Path,
    brand_text: str,
    cta_text: str,
    logo_url: Optional[str],
    job_id: str,
):
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    brand_file = JOBS_DIR / f"{job_id}-brand.txt"
    cta_file = JOBS_DIR / f"{job_id}-cta.txt"
    brand_file.write_text(brand_text, encoding="utf-8")
    cta_file.write_text(cta_text, encoding="utf-8")
    logo_path = SOURCES_DIR / f"{job_id}-logo"

    top_text = (
        f"drawtext=fontfile={font}:textfile={brand_file}:"
        "fontcolor=white:fontsize=h/12:x=24:y=24:"
        "box=1:boxcolor=black@0.45:boxborderw=10"
    )
    cta_filter = (
        f"drawtext=fontfile={font}:textfile={cta_file}:"
        "fontcolor=white:fontsize=h/22:x=(w-text_w)/2:y=h-text_h-30:"
        "box=1:boxcolor=black@0.55:boxborderw=10"
    )
    filters = ",".join([top_text, cta_filter]) if cta_text else top_text

    try:
        if logo_url:
            _download_logo(logo_url, logo_path)
            complex_filter = (
                "[1:v]scale=160:-1[logo];"
                "[0:v][logo]overlay=W-w-24:24[base];"
                f"[base]{filters}[v]"
            )
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(source), "-i", str(logo_path),
                "-filter_complex", complex_filter,
                "-map", "[v]", "-map", "0:a?",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                str(output),
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(source),
                "-vf", filters,
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                str(output),
            ]
        subprocess.run(cmd, check=True)
    finally:
        brand_file.unlink(missing_ok=True)
        cta_file.unlink(missing_ok=True)
        logo_path.unlink(missing_ok=True)


def ensure_promo_avatar(payload: dict):
    """Resolve either the bundled animated 3D presenter or a user-supplied MP4."""
    avatar_id = clean_avatar_id(payload["avatar_id"])
    avatar_url = payload.get("avatar_url")

    if avatar_id in avatar_cache or avatar_prepared(avatar_id) or avatar_url:
        return ensure_avatar(avatar_id, avatar_url)

    if avatar_id == FAST_PROMO_AVATAR_ID and FAST_PROMO_AVATAR_PATH.is_file():
        return prepare_avatar_from_local(avatar_id, FAST_PROMO_AVATAR_PATH)

    raise RuntimeError(
        "No hay avatar rápido disponible. Agrega el video 3D incluido o proporciona una URL MP4 directa."
    )


def run_promo_job(job_id: str, payload: dict):
    """Fast promo path for RTX 3060: animated MP4 + local voice + MuseTalk.

    Wan2.1 is intentionally excluded from this interactive route. The fixed,
    animated 3D source (or an uploaded MP4 URL) keeps jobs practical on 12 GB
    VRAM while MuseTalk provides the lip synchronization.
    """
    set_job(job_id, status="running", stage="preparing_avatar")
    wav_path = JOBS_DIR / f"{job_id}.wav"
    lip_path = JOBS_DIR / f"{job_id}-lipsync.mp4"
    final_path = JOBS_DIR / f"{job_id}.mp4"

    try:
        init_engine()
        avatar = ensure_promo_avatar(payload)

        set_job(job_id, stage="synthesizing_voice")
        length_scale = (
            FAST_VOICE_LENGTH_SCALE
            if payload.get("voice_speed", "fast") == "fast"
            else NORMAL_VOICE_LENGTH_SCALE
        )
        synthesize_speech(payload["text"], wav_path, length_scale=length_scale)

        set_job(job_id, stage="syncing_avatar")
        avatar.inference(
            audio_path=str(wav_path),
            out_vid_name=job_id,
            fps=FPS,
            skip_save_images=False,
        )
        produced = avatar_base(payload["avatar_id"]) / "vid_output" / f"{job_id}.mp4"
        if not produced.is_file() or not produced.stat().st_size:
            raise RuntimeError("MuseTalk no generó el MP4 promocional esperado")
        shutil.copy2(produced, lip_path)

        set_job(job_id, stage="adding_brand")
        apply_branding(
            source=lip_path,
            output=final_path,
            brand_text=payload["brand_text"],
            cta_text=payload["cta_text"],
            logo_url=payload.get("logo_url"),
            job_id=job_id,
        )
        set_job(
            job_id,
            status="done",
            stage="done",
            finished_at=time.time(),
            video_path=str(final_path),
            video_endpoint=f"/jobs/{job_id}/video",
        )
        cleanup_outputs()
    except Exception as exc:
        set_job(
            job_id,
            status="error",
            stage="error",
            finished_at=time.time(),
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        wav_path.unlink(missing_ok=True)
        lip_path.unlink(missing_ok=True)


def cleanup_outputs(keep: int = 30):
    files = sorted(JOBS_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[keep:]:
        old.unlink(missing_ok=True)


def run_one_job(job_id: str, payload: dict):
    set_job(job_id, status="running", started_at=time.time())
    wav_path = JOBS_DIR / f"{job_id}.wav"
    final_path = JOBS_DIR / f"{job_id}.mp4"

    try:
        set_job(job_id, stage="loading_lipsync")
        init_engine()
        set_job(job_id, stage="synthesizing_speech")
        synthesize_speech(payload["text"], wav_path)
        avatar = ensure_avatar(payload["avatar_id"], payload.get("avatar_url"))
        set_job(job_id, stage="syncing_lips")

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


def run_motion_job(job_id: str, payload: dict):
    """Generate body motion first; MuseTalk must receive the moving MP4."""
    work = Path(payload["work_dir"])
    wav_path = work / "speech.wav"
    final_path = JOBS_DIR / f"{job_id}.mp4"
    avatar_id = f"motion-{job_id[:16]}"
    set_job(job_id, status="running", stage="generating_body_motion", started_at=time.time())
    try:
        release_musetalk_engine()
        moving_video = run_motion(work, JOBS_DIR / f"{job_id}-motion.log")
        set_job(job_id, stage="loading_lipsync")
        avatar = prepare_avatar_from_local(avatar_id, moving_video)
        set_job(job_id, stage="synthesizing_speech")
        synthesize_speech(payload["text"], wav_path)
        set_job(job_id, stage="syncing_lips")
        avatar.inference(audio_path=str(wav_path), out_vid_name=job_id,
                         fps=FPS, skip_save_images=False)
        produced = avatar_base(avatar_id) / "vid_output" / f"{job_id}.mp4"
        if not produced.is_file() or produced.stat().st_size == 0:
            raise RuntimeError("MuseTalk no generó el video con movimiento y voz")
        shutil.copy2(produced, final_path)
        set_job(job_id, status="done", stage="done", finished_at=time.time(),
                video_path=str(final_path), video_endpoint=f"/jobs/{job_id}/video")
        cleanup_outputs()
    except Exception as exc:
        set_job(job_id, status="error", stage="error", finished_at=time.time(),
                error=f"{type(exc).__name__}: {exc}")
    finally:
        avatar_cache.pop(avatar_id, None)
        shutil.rmtree(avatar_base(avatar_id), ignore_errors=True)
        (SOURCES_DIR / f"{avatar_id}.mp4").unlink(missing_ok=True)
        shutil.rmtree(work, ignore_errors=True)


def worker_loop():
    while True:
        job_id, payload = job_queue.get()
        try:
            if payload.get("kind") == "motion":
                run_motion_job(job_id, payload)
            elif payload.get("kind") == "promo":
                run_promo_job(job_id, payload)
            else:
                run_one_job(job_id, payload)
        finally:
            job_queue.task_done()


@app.on_event("startup")
def startup():
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
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "queued_jobs": job_queue.qsize(),
        "musetalk_loaded": engine_ready,
        "wan_model_ready": wan_model_ready(),
        "motion_installed": motion_available(),
        "fast_promo_avatar_source": FAST_PROMO_AVATAR_PATH.is_file(),
        "fast_promo_avatar_prepared": avatar_prepared(FAST_PROMO_AVATAR_ID),
    }


@app.post("/motion-jobs", status_code=202, dependencies=[Depends(require_auth)])
def create_motion_job(
    text: str = Form(..., min_length=1, max_length=800),
    avatar: UploadFile = File(...),
    reference: UploadFile = File(...),
):
    if not text.strip():
        raise HTTPException(status_code=422, detail="Escribe lo que debe decir el avatar")
    if not motion_available():
        raise HTTPException(status_code=503, detail="El motor de movimiento todavía no está instalado")
    job_id = uuid.uuid4().hex
    work = SOURCES_DIR / f"motion-{job_id}"
    work.mkdir()
    try:
        save_upload(avatar.file, work / "avatar.source", 20 * 1024 * 1024)
        save_upload(reference.file, work / "reference.source", 100 * 1024 * 1024)
    except ValueError as exc:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(status_code=413, detail=str(exc))
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise
    finally:
        avatar.file.close()
        reference.file.close()
    set_job(job_id, id=job_id, kind="motion", status="queued", stage="queued",
            created_at=time.time())
    job_queue.put((job_id, {"kind": "motion", "text": text.strip(), "work_dir": str(work)}))
    return {"id": job_id, "status": "queued", "status_endpoint": f"/jobs/{job_id}",
            "video_endpoint": f"/jobs/{job_id}/video"}


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
        "kind": "avatar",
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


def parse_studio_message(message: str):
    raw = message.strip()
    lower = raw.lower()

    orientation = "vertical"
    if any(word in lower for word in ("horizontal", "landscape", "youtube", "16:9")):
        orientation = "landscape"

    speech = None
    quoted = re.findall(r'["“”](.+?)["“”]', raw)
    if quoted:
        speech = quoted[-1].strip()

    if not speech:
        patterns = [
            r"(?:que diga|diciendo|y diga|debe decir)\s*[:：-]?\s*(.+)$",
            r"(?:texto|mensaje)\s*[:：-]\s*(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, raw, flags=re.IGNORECASE | re.DOTALL)
            if match:
                speech = match.group(1).strip().strip('"“”')
                break

    if not speech:
        raise HTTPException(
            status_code=422,
            detail=(
                "Indica lo que debe decir la persona. Ejemplo: "
                "Crea una mujer joven tipo influencer promocionando AZTV "
                "que diga \"Descarga AZTV hoy\"."
            ),
        )

    person_prompt = raw
    for marker in (" que diga ", " diciendo ", " y diga ", " debe decir "):
        idx = lower.find(marker)
        if idx >= 0:
            person_prompt = raw[:idx].strip(" ,.-")
            break

    if quoted:
        person_prompt = person_prompt.replace(f'"{speech}"', "").strip(" ,.-")

    brand = "AZTV"
    brand_match = re.search(
        r"(?:marca|brand)\s+([A-Za-z0-9_-]{2,30})",
        raw,
        flags=re.IGNORECASE,
    )
    if brand_match:
        brand = brand_match.group(1)

    cta = "Descárgala hoy"
    if "sin texto" in lower or "sin letras" in lower:
        cta = ""

    return {
        "text": speech,
        "person_prompt": person_prompt,
        "brand_text": brand,
        "cta_text": cta,
        "logo_url": None,
        "orientation": orientation,
        "steps": 28,
        "seed": -1,
    }


@app.get("/studio", response_class=HTMLResponse)
def studio(
    access_key: Optional[str] = Query(default=None, alias="access"),
    studio_session: Optional[str] = Cookie(default=None, alias=STUDIO_SESSION_COOKIE),
):
    """Open Studio only through a private link, then remove its key from the URL."""
    if STUDIO_ACCESS_KEY:
        if access_key and hmac.compare_digest(access_key, STUDIO_ACCESS_KEY):
            response = RedirectResponse(url="/studio", status_code=303)
            response.set_cookie(
                key=STUDIO_SESSION_COOKIE,
                value=STUDIO_ACCESS_KEY,
                max_age=STUDIO_SESSION_MAX_AGE_SECONDS,
                secure=True,
                httponly=True,
                samesite="strict",
                path="/",
            )
            return response
        if not has_studio_session(studio_session):
            raise HTTPException(status_code=401, detail="Abre tu enlace privado del Studio.")
    return HTMLResponse(Path(__file__).with_name("studio.html").read_text(encoding="utf-8"))


@app.post("/studio/request", status_code=202, dependencies=[Depends(require_auth)])
def studio_request(request: StudioRequest):
    parsed = parse_studio_message(request.message)
    job_id = uuid.uuid4().hex
    avatar_url = str(request.avatar_url) if request.avatar_url else None
    payload = {
        "kind": "promo",
        **parsed,
        "avatar_id": FAST_PROMO_AVATAR_ID if not avatar_url else f"custom-{job_id[:16]}",
        "avatar_url": avatar_url,
        "brand_text": request.brand_text.strip(),
        "cta_text": request.cta_text.strip(),
        "voice_speed": request.voice_speed,
    }
    set_job(
        job_id,
        id=job_id,
        kind="promo",
        status="queued",
        stage="queued",
        created_at=time.time(),
        brand_text=payload["brand_text"],
        studio_message=request.message.strip(),
    )
    job_queue.put((job_id, payload))
    return {
        "id": job_id,
        "status": "queued",
        "status_endpoint": f"/jobs/{job_id}",
        "video_endpoint": f"/jobs/{job_id}/video",
    }


@app.post("/promos", status_code=202, dependencies=[Depends(require_auth)])
def create_promo(request: PromoRequest):
    job_id = uuid.uuid4().hex
    payload = {
        "kind": "promo",
        "text": request.text.strip(),
        "person_prompt": request.person_prompt.strip(),
        "avatar_id": FAST_PROMO_AVATAR_ID,
        "avatar_url": None,
        "brand_text": request.brand_text.strip(),
        "cta_text": request.cta_text.strip(),
        "logo_url": str(request.logo_url) if request.logo_url else None,
        "orientation": request.orientation,
        "voice_speed": "fast",
        "steps": request.steps,
        "seed": request.seed,
    }
    set_job(
        job_id,
        id=job_id,
        kind="promo",
        status="queued",
        stage="queued",
        created_at=time.time(),
        brand_text=request.brand_text.strip(),
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
