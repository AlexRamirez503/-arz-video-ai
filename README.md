# ARZ Video AI

Generador propio de videos de personas hablando, preparado para desplegarse con GPU NVIDIA en SaladCloud.

## Qué incluye

- Wan2.1 T2V-1.3B para generar desde cero un presentador en movimiento.
- MuseTalk 1.5 para sincronización de labios.
- Piper TTS con voz `es_MX-ald-medium` para español.
- Branding final con FFmpeg (texto AZTV, CTA y logo opcional).
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
WAN_MODEL_DIR=/data/models/Wan2.1-T2V-1.3B
```

La primera solicitud a `POST /promos` descarga los pesos de Wan2.1 a
`WAN_MODEL_DIR`. Reserva al menos 12 GB libres adicionales en el almacenamiento
del contenedor. Las solicitudes posteriores reutilizan esos pesos.

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

### Crear promoción desde cero

`POST /promos`

Ejemplo:

```json
{
  "text": "Descarga AZTV y disfruta entretenimiento donde quieras.",
  "person_prompt": "presentador joven latino, sonriente, ropa casual moderna",
  "brand_text": "AZTV",
  "cta_text": "Descárgala hoy",
  "orientation": "vertical",
  "steps": 28
}
```

Flujo automático: Wan2.1 genera una persona en movimiento → Piper crea la voz →
MuseTalk sincroniza los labios → FFmpeg agrega la marca y el CTA. Para usar un
logo real, agrega `"logo_url": "https://.../logo.png"`.

El progreso aparece en `stage`, por ejemplo
`downloading_video_model`, `generating_person_video`, `syncing_lips` y
`adding_brand`.

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
- Almacenamiento recomendado con Wan2.1: 40 GB o más
- Wan2.1 1.3B se ejecuta de forma secuencial con MuseTalk para compartir una sola GPU

## Conexión con ChatGPT

FastAPI expone automáticamente `/openapi.json`. Después de que el servidor esté funcionando, se puede colocar un pequeño servidor MCP delante de esta API para que ChatGPT cree trabajos, consulte su estado y recupere el video terminado.
