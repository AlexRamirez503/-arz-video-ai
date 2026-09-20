# ARZ Video AI

**ARZ Video AI** is a fast Spanish-language promotional-video generator designed for a single NVIDIA RTX 3060 on SaladCloud. The default experience is a chat-style studio at `/studio`: write the message, choose the brand and CTA, and the service produces an MP4 with an animated 3D presenter, local Spanish voice, lip synchronization, and branded overlays.

## Fast mode

The interactive chat deliberately **does not run Wan2.1 text-to-video**. Generating a new video character with Wan2.1 is too slow and memory-intensive for a responsive RTX 3060 workflow. Instead, the image includes an eight-second animated 3D presenter source clip. MuseTalk synchronizes the presenter to the requested narration, and Piper creates the voice locally.

A user can also provide a direct URL to a custom animated MP4 in the studio. The first use prepares that avatar; subsequent requests reuse the prepared cache. Jobs are serialized so the single GPU is not overloaded.

| Capability | Fast chat mode |
|---|---|
| Natural-language request | Yes |
| Default animated 3D presenter | Yes |
| Spanish local voice | Yes, fast or normal pacing |
| Lip synchronization | Yes, via MuseTalk 1.5 |
| Brand and CTA overlay | Yes |
| Custom animated avatar | Yes, with a direct MP4 URL |
| Wan2.1 generation on chat requests | No |

## Public studio URL

Configure HTTP networking on port `8000` in SaladCloud. The Container Gateway creates a domain for the container group. Open:

```text
https://YOUR-SALAD-GATEWAY-DOMAIN/studio
```

If `API_TOKEN` is set in SaladCloud, paste it once in the Studio. The token remains only in the browser's local storage.

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
```

`FAST_PROMO_AVATAR_PATH` defaults to `/app/assets/default_3d_presenter.mp4`, which is bundled into the image. It can be overridden only when supplying a compatible replacement source in a custom image.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | GPU, queue, and fast-avatar readiness |
| `GET` | `/studio` | Chat-style web interface |
| `POST` | `/studio/request` | Create a fast promotion from a message |
| `POST` | `/promos` | Create a fast promotion programmatically |
| `POST` | `/jobs` | Lip-sync a supplied avatar URL |
| `GET` | `/jobs/{job_id}` | Retrieve job status |
| `GET` | `/jobs/{job_id}/video` | Download completed MP4 |

Example request:

```json
{
  "message": "Promociona AZTV y di \"Disfruta tus canales favoritos donde quieras. Descarga AZTV hoy.\"",
  "brand_text": "AZTV",
  "cta_text": "Descárgala hoy",
  "voice_speed": "fast"
}
```

To use a custom animated avatar, add `"avatar_url": "https://example.com/animated-avatar.mp4"`. The URL must point directly to an MP4 file and the first preparation may take longer than later videos.

## Development checks

```bash
python -m py_compile server/main.py
python -m unittest discover -s tests -v
```

The checks validate job orchestration and the fast path. They do not substitute for a GPU run with the MuseTalk weights.
