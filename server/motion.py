"""Bounded image + reference-video preparation and isolated MimicMotion process."""
import os
import subprocess
from pathlib import Path

MOTION_HOME = Path(os.getenv("MOTION_HOME", "/opt/MimicMotion"))
MOTION_PYTHON = Path(os.getenv("MOTION_PYTHON", "/opt/motion-venv/bin/python"))
RUNNER = Path(__file__).with_name("motion_inference.py")


def motion_available():
    # Installation check only. Weights are downloaded on the first motion job.
    return MOTION_PYTHON.is_file() and (MOTION_HOME / "inference.py").is_file()


def save_upload(stream, target, limit):
    size = 0
    with target.open("wb") as output:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                raise ValueError(f"El archivo supera el límite de {limit // 1024 // 1024} MB")
            output.write(chunk)
    if not size:
        raise ValueError("El archivo está vacío")


def run_motion(work, log_path):
    import cv2
    if not motion_available():
        raise RuntimeError("El motor de movimiento no está instalado")
    image = cv2.imread(str(work / "avatar.source"))
    if image is None:
        raise ValueError("El avatar debe ser una imagen PNG o JPG válida")
    if not cv2.imwrite(str(work / "avatar.png"), image):
        raise RuntimeError("No se pudo preparar la imagen del avatar")
    reference = work / "reference.mp4"
    # Keep the driving clip and generated output at the same 15 fps. The UI
    # explicitly describes the 8-second reference window, matching MuseTalk.
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i",
        str(work / "reference.source"), "-t", "8", "-an", "-vf",
        "fps=15,scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", str(reference),
    ], check=True, timeout=120)
    cap = cv2.VideoCapture(str(reference))
    try:
        if not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_COUNT) < 30:
            raise ValueError("El video de referencia debe tener al menos 2 segundos")
    finally:
        cap.release()
    output = work / "moving-avatar.mp4"
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    env["OMP_NUM_THREADS"] = "4"
    # A separate process releases all movement model VRAM before lip sync.
    with log_path.open("w") as log:
        result = subprocess.run([
            str(MOTION_PYTHON), str(RUNNER), "--image", str(work / "avatar.png"),
            "--reference", str(reference), "--output", str(output),
        ], cwd=MOTION_HOME, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=5400)
    if result.returncode != 0 or not output.is_file() or not output.stat().st_size:
        raise RuntimeError("Falló la generación de movimiento; revisa el registro del trabajo. "
                           "No se ha sustituido el resultado por una imagen inmóvil.")
    cap = cv2.VideoCapture(str(output))
    try:
        if not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_COUNT) < 2:
            raise RuntimeError("El motor de movimiento no devolvió un video válido")
    finally:
        cap.release()
    return output
