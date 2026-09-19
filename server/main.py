import gc
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
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field, HttpUrl

MUSETALK_HOME = Path(os.getenv("MUSETALK_HOME", "/opt/MuseTalk")).resolve()
DATA_DIR = Path(os.getenv("DATA_DIR", "/data")).resolve()
VOICE_MODEL = Path(os.getenv("PIPER_VOICE", "/opt/voices/es_MX-ald-medium.onnx"))
API_TOKEN = os.getenv("API_TOKEN", "")
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


def synthesize_speech(text: str, output_wav: Path):
    cmd = [
        "piper",
        "--model", str(VOICE_MODEL),
        "--output_file", str(output_wav),
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


def run_promo_job(job_id: str, payload: dict):
    set_job(job_id, status="running", stage="preparing")
    wav_path = JOBS_DIR / f"{job_id}.wav"
    lip_path = JOBS_DIR / f"{job_id}-lipsync.mp4"
    final_path = JOBS_DIR / f"{job_id}.mp4"
    wan_video = None

    try:
        wan_video = run_wan_video(
            job_id=job_id,
            person_prompt=payload["person_prompt"],
            orientation=payload["orientation"],
            steps=payload["steps"],
            seed=payload["seed"],
        )

        set_job(job_id, stage="loading_lipsync")
        avatar_id = f"promo-{job_id[:16]}"
        avatar = prepare_avatar_from_local(avatar_id, wan_video)

        set_job(job_id, stage="synthesizing_speech")
        synthesize_speech(payload["text"], wav_path)

        set_job(job_id, stage="syncing_lips")
        avatar.inference(
            audio_path=str(wav_path),
            out_vid_name=job_id,
            fps=FPS,
            skip_save_images=False,
        )
        produced = avatar_base(avatar_id) / "vid_output" / f"{job_id}.mp4"
        if not produced.exists():
            raise RuntimeError("MuseTalk did not create the expected promo MP4")
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
        if wan_video is not None:
            wan_video.unlink(missing_ok=True)


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


def worker_loop():
    while True:
        job_id, payload = job_queue.get()
        try:
            if payload.get("kind") == "promo":
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
def studio():
    return HTMLResponse(
        """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>AZTV AI Studio</title>
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#09090b;color:#fff;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif}
.wrap{max-width:760px;margin:auto;min-height:100vh;padding:24px 14px 120px}
h1{font-size:30px;margin:8px 4px 4px}
.sub{color:#9ca3af;margin:0 4px 24px;line-height:1.45}
.card{background:#151519;border:1px solid #2b2b31;border-radius:20px;padding:16px;margin-bottom:14px}
label{display:block;color:#b7bac2;font-size:13px;margin:0 0 8px}
input,textarea{width:100%;border:1px solid #33343b;background:#0e0e11;color:white;border-radius:14px;padding:13px;font-size:16px}
textarea{min-height:128px;resize:vertical;line-height:1.4}
button{border:0;border-radius:14px;padding:14px 18px;font-size:16px;font-weight:800;background:white;color:#050505;width:100%}
button:disabled{opacity:.45}
.example{color:#9ca3af;font-size:13px;line-height:1.5;margin-top:10px}
.msg{border-radius:18px;padding:14px;margin:12px 0;background:#17171c;border:1px solid #2b2b31}
.mine{background:#202838}
.badge{display:inline-block;border-radius:999px;padding:5px 9px;font-size:11px;font-weight:800;margin-bottom:8px}
.queued{background:#233452;color:#93c5fd}.running{background:#493b16;color:#fde68a}.done{background:#173d27;color:#86efac}.error{background:#4a1d24;color:#fda4af}
video{width:100%;border-radius:14px;background:#000;margin-top:10px}
a.download{display:block;background:#fff;color:#000;text-decoration:none;text-align:center;padding:12px;border-radius:12px;font-weight:800;margin-top:10px}
.small{font-size:12px;color:#8a8d96;word-break:break-all}
#notice{color:#fbbf24;font-size:13px;margin-top:10px;min-height:18px}
</style>
</head>
<body><div class="wrap">
<h1>AZTV AI Studio</h1>
<p class="sub">Escribe lo que quieres como un mensaje. El sistema genera la persona, la voz, sincroniza labios y prepara el video.</p>

<div class="card">
<label>API Token</label>
<input id="token" type="password" placeholder="Pega tu token una sola vez">
<div class="example">Se guarda únicamente en este navegador.</div>
</div>

<div class="card">
<label>¿Qué quieres crear?</label>
<textarea id="message" placeholder='Ejemplo: Crea una mujer joven tipo influencer, sonriente y moviendo las manos, promocionando AZTV, que diga "Descarga AZTV y disfruta entretenimiento donde quieras."'></textarea>
<div class="example">Para indicar la voz exacta usa “que diga ...” o pon el texto entre comillas.</div>
<div id="notice"></div>
<button id="send">Generar promoción</button>
</div>

<div id="chat"></div>
</div>
<script>
const tokenEl=document.getElementById('token');
const msgEl=document.getElementById('message');
const sendEl=document.getElementById('send');
const chat=document.getElementById('chat');
const notice=document.getElementById('notice');
tokenEl.value=localStorage.getItem('aztv_api_token')||'';
tokenEl.addEventListener('change',()=>localStorage.setItem('aztv_api_token',tokenEl.value.trim()));
let jobs=JSON.parse(localStorage.getItem('aztv_studio_jobs')||'[]');

function headers(){return {'Authorization':'Bearer '+tokenEl.value.trim(),'Content-Type':'application/json'};}
function save(){localStorage.setItem('aztv_studio_jobs',JSON.stringify(jobs.slice(0,30)));}

function render(){
  chat.innerHTML='';
  jobs.forEach(j=>{
    const el=document.createElement('div'); el.className='msg';
    let cls=j.status==='done'?'done':j.status==='error'?'error':j.status==='running'?'running':'queued';
    let media='';
    if(j.status==='done') media='<video controls playsinline src="/jobs/'+j.id+'/video"></video><a class="download" href="/jobs/'+j.id+'/video" target="_blank">Abrir / descargar MP4</a>';
    if(j.status==='error') media='<div style="color:#fca5a5;margin-top:8px">'+(j.error||'Error')+'</div>';
    el.innerHTML='<span class="badge '+cls+'">'+(j.stage||j.status||'queued')+'</span><div>'+escapeHtml(j.message)+'</div>'+media+'<div class="small">'+j.id+'</div>';
    chat.appendChild(el);
  });
}
function escapeHtml(s){return (s||'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));}

async function poll(){
  for(const j of jobs){
    if(!j.id || j.status==='done' || j.status==='error') continue;
    try{
      const r=await fetch('/jobs/'+j.id,{headers:{'Authorization':'Bearer '+tokenEl.value.trim()}});
      if(!r.ok) continue;
      const d=await r.json();
      j.status=d.status||j.status; j.stage=d.stage||d.status||j.stage; j.error=d.error||'';
    }catch(e){}
  }
  save(); render();
}
setInterval(poll,6000);

sendEl.onclick=async()=>{
  const token=tokenEl.value.trim(), message=msgEl.value.trim();
  if(!token){notice.textContent='Primero pega tu API Token.';return;}
  if(!message){notice.textContent='Escribe lo que quieres crear.';return;}
  localStorage.setItem('aztv_api_token',token);
  sendEl.disabled=true; notice.textContent='Enviando...';
  try{
    const r=await fetch('/studio/request',{method:'POST',headers:headers(),body:JSON.stringify({message})});
    const d=await r.json();
    if(!r.ok) throw new Error(d.detail||'No se pudo crear');
    jobs.unshift({id:d.id,message,status:d.status||'queued',stage:'queued'});
    save(); render(); msgEl.value=''; notice.textContent='Trabajo enviado. Puedes dejar esta página abierta.';
  }catch(e){notice.textContent=e.message;}
  finally{sendEl.disabled=false;}
};
render(); poll();
</script>
</body></html>"""
    )


@app.post("/studio/request", status_code=202, dependencies=[Depends(require_auth)])
def studio_request(request: StudioRequest):
    parsed = parse_studio_message(request.message)
    job_id = uuid.uuid4().hex
    payload = {"kind": "promo", **parsed}
    set_job(
        job_id,
        id=job_id,
        kind="promo",
        status="queued",
        stage="queued",
        created_at=time.time(),
        brand_text=parsed["brand_text"],
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
        "brand_text": request.brand_text.strip(),
        "cta_text": request.cta_text.strip(),
        "logo_url": str(request.logo_url) if request.logo_url else None,
        "orientation": request.orientation,
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
