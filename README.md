# ARZ Video AI

**ARZ Video AI** is a Spanish-language video generator designed for a single NVIDIA RTX 3060 on SaladCloud. The Studio at `/studio` offers two separate modes: create a free-form clip from a description, or create a branded MP4 with an animated 3D presenter, local Spanish voice, lip synchronization, and overlays.

## Studio modes

The **Create any video** mode uses local Wan2.1 T2V-1.3B to turn an unrestricted description into a short MP4. It accepts requests such as a child riding a bicycle, a singing house, or animated-cartoon scenes. It runs locally on the RTX 3060 with CPU/model offloading, so jobs are serialized and may take several minutes. The model weights are downloaded to `/data/models` on the first request, which makes that first job longer and requires at least 12 GB of free persistent disk space.

The **Avatar 3D speaking** mode remains optimized for responsive promotions. It uses the bundled eight-second animated 3D presenter source clip, Piper for local Spanish voice, and MuseTalk for lip synchronization. The avatar reads exactly the text entered in its script box; masculine voice and normal pace are the defaults.

A user can also provide a direct URL to a custom animated MP4 in the studio. The first use prepares that avatar; subsequent requests reuse the prepared cache. Jobs are serialized so the single GPU is not overloaded.

| Capability | Create any video | Avatar 3D speaking |
|---|---|
| Unrestricted description | Yes | Script only |
| Examples | House singing, child on a bicycle, cartoons | Product presenter or custom MP4 avatar |
| Default animated 3D presenter | No | Yes |
| Spanish local voice | No | Yes, masculine or feminine |
| Lip synchronization | No | Yes, via MuseTalk 1.5 |
| Brand and CTA overlay | No | Yes |
| Custom animated avatar | No | Yes, with a direct MP4 URL |
| Local engine | Wan2.1 T2V-1.3B | MuseTalk 1.5 + Piper |
| Expected speed on RTX 3060 | Slow, several minutes | Faster |

## Public studio URL

Configure HTTP networking on port `8000` in SaladCloud. The Container Gateway creates a domain for the container group. Open:

```text
https://YOUR-SALAD-GATEWAY-DOMAIN/studio
```

Set `STUDIO_ACCESS_KEY` in SaladCloud and open the private `/studio?access=...` link once. Safari keeps the access value within the Studio tab and removes it from the visible address bar. API requests remain protected without asking the user to paste a token.

## Image for SaladCloud

GitHub Actions publishes the image after every push to `main`:

```text
ghcr.io/alexramirez503/arz-video-ai:latest
```

In SaladCloud, deploy that image with a single replica and HTTP networking configured for container port `8000`. Recreate the instance after GitHub Actions completes so Salad pulls the new image.

## Environment variables

```text
API_TOKEN=<optional-secret>
MUSETALK_BATCH_SIZE=8
MUSETALK_FPS=25
API_PORT=8000
FAST_VOICE_LENGTH_SCALE=0.84
NORMAL_VOICE_LENGTH_SCALE=1.0
WAN_FREE_VIDEO_STEPS=24
WAN_FREE_VIDEO_FRAME_COUNT=81
```

`FAST_PROMO_AVATAR_PATH` defaults to `/app/assets/default_3d_presenter.mp4`, which is bundled into the image. It can be overridden only when supplying a compatible replacement source in a custom image.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | GPU, queue, and fast-avatar readiness |
| `GET` | `/studio` | Chat-style web interface |
| `POST` | `/studio/request` | Create a free-form local video or an avatar video from Studio input |
| `POST` | `/promos` | Create a fast promotion programmatically |
| `POST` | `/jobs` | Lip-sync a supplied avatar URL |
| `GET` | `/jobs/{job_id}` | Retrieve job status |
| `GET` | `/jobs/{job_id}/video` | Download completed MP4 |

Example request:

```json
{
  "message": "Una casa colorida canta bajo la lluvia, estilo dibujos animados.",
  "mode": "free_video",
  "free_style": "cartoon",
  "orientation": "vertical"
}
```

To use a custom animated avatar, add `"avatar_url": "https://example.com/animated-avatar.mp4"`. The URL must point directly to an MP4 file and the first preparation may take longer than later videos.

## Development checks

```bash
python -m py_compile server/main.py
python -m unittest discover -s tests -v
```

The checks validate job orchestration and the fast path. They do not substitute for a GPU run with the MuseTalk weights.
