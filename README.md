# ARZ Video AI

Generador propio de videos de personas hablando, preparado para desplegarse con GPU NVIDIA en SaladCloud.

## Qué incluye

- MuseTalk 1.5 para sincronización de labios.
- Piper TTS con voz `es_MX-ald-medium` para español.
- FastAPI para controlar el generador por API.
- Cola de trabajos de una sola GPU para evitar saturar la RTX 3060.
- Caché de avatares: la primera preparación tarda más; los siguientes videos reutilizan el avatar.
- Imagen Docker publicada automáticamente en GitHub Container Registry.

## Imagen para SaladCloud

Cuando GitHub Actions termine correctamente, usa:

```
ghcr.io/alexramirez503/arz-video-ai:latest
```

La imagen se publica con el nombre limpio `arz-video-ai` aunque este repositorio haya sido creado con un guion inicial.

## Variables recomendadas en SaladCloud

```
API_TOKEN=<un-token-secreto-largo>
MUSETALK_BATCH_SIZE=8
MUSETALK_FPS=25
API_PORT=8000
```

## API

### Estado

`GET /health`

### Crear video

`POST /jobs`

Ejemplo JSON:

```json
{
  "text": "Cristo te ama y tiene un propósito para tu vida.",
  "avatar_id": "presentador-1",
  "avatar_url": "https://example.com/avatar.mp4"
}
```

La primera vez que se usa un `avatar_id`, se envía `avatar_url`. En solicitudes posteriores se puede omitir para reutilizar el avatar preparado.

### Revisar trabajo

`GET /jobs/{job_id}`

### Descargar resultado

`GET /jobs/{job_id}/video`

Si configuras `API_TOKEN`, usa:

```
Authorization: Bearer TU_TOKEN
```

## SaladCloud recomendado para este proyecto

- GPU: RTX 3060 12 GB
- vCPU: 4
- RAM: 24 GB
- Réplicas: 1
- Puerto del contenedor: 8000

## Conexión con ChatGPT

FastAPI expone automáticamente `/openapi.json`. Después de que el servidor esté funcionando, se puede colocar un pequeño servidor MCP delante de esta API para que ChatGPT cree trabajos, consulte su estado y recupere el video terminado.
