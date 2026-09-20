"""Run in the isolated motion environment, never inside MuseTalk's process."""
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    home = Path(os.getenv("MOTION_HOME", "/opt/MimicMotion"))
    os.chdir(home)
    sys.path.insert(0, str(home))

    from huggingface_hub import hf_hub_download
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("MimicMotion necesita una GPU CUDA disponible")
    # Official weights only; downloads are cached and reused.
    for name in ("yolox_l.onnx", "dw-ll_ucoco_384.onnx"):
        hf_hub_download("yzd-v/DWPose", name, local_dir=str(home / "models/DWPose"))
    checkpoint = hf_hub_download("tencent/MimicMotion", "MimicMotion_1.pth",
                                local_dir=str(home / "models"))
    # The original 16-frame checkpoint, smaller resolution and VAE chunks
    # reduce peak memory for the 12 GB GPU. GPU validation is still required.
    torch.set_default_dtype(torch.float16)
    from inference import preprocess
    from mimicmotion.dwpose.dwpose_detector import dwpose_detector
    from mimicmotion.utils.loader import create_pipeline
    from mimicmotion.utils.utils import save_to_mp4
    from torchvision.transforms.functional import to_pil_image

    # CPU ONNX pose extraction avoids an additional CUDA/cuDNN dependency and
    # leaves GPU memory to the video model. The detector initializes lazily.
    det_path, pose_path, _ = dwpose_detector.args
    dwpose_detector.args = (det_path, pose_path, "cpu")

    device = torch.device("cuda:0")
    pose, image = preprocess(args.reference, args.image, resolution=320, sample_stride=1)
    if pose.size(0) < 16:
        raise RuntimeError("No hay suficientes fotogramas de referencia")
    config = SimpleNamespace(
        base_model_path="stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        ckpt_path=checkpoint,
    )
    pipeline = create_pipeline(config, device)
    images = [to_pil_image(im.to(torch.uint8)) for im in (image + 1) * 127.5]
    with torch.inference_mode():
        frames = pipeline(
            images, image_pose=pose, num_frames=pose.size(0), tile_size=16, tile_overlap=6,
            height=pose.shape[-2], width=pose.shape[-1], fps=7, noise_aug_strength=0,
            num_inference_steps=25, generator=torch.Generator(device=device).manual_seed(42),
            min_guidance_scale=2.0, max_guidance_scale=2.0, decode_chunk_size=2,
            output_type="pt", device=device,
        ).frames.cpu()
    save_to_mp4((frames[0, 1:] * 255).to(torch.uint8), args.output, fps=15)


if __name__ == "__main__":
    main()
