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

### Panel tipo chat

Abre `GET /studio` en el navegador. Pega tu `API_TOKEN` una sola vez y después
puedes escribir órdenes normales como:

> Crea una mujer joven tipo influencer, sonriente y moviendo las manos,
> promocionando AZTV, que diga "Descarga AZTV y disfruta entretenimiento donde quieras."

El panel crea el trabajo, muestra el progreso y presenta el MP4 cuando termina.
También existe `POST /studio/request` para enviar una sola instrucción en lenguaje
natural.

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
# Movimiento del avatar desde un video de referencia (pendiente de prueba GPU)

La nueva sección de `/studio` recibe una foto PNG/JPG, un video de referencia
y el texto hablado. `POST /motion-jobs` recibe los campos multipart `avatar`,
`reference` y `text`, con la misma autenticación de los demás trabajos.

El flujo es foto + referencia → MimicMotion → video corporal → Piper →
MuseTalk → MP4. MuseTalk recibe el video generado, nunca la foto original como
sustitución silenciosa si falla MimicMotion. Cada trabajo usa un avatar temporal
distinto para evitar reutilizar un avatar inmóvil de la caché.

La referencia debe durar al menos 2 segundos; se usan los primeros 8, a 15 fps.
Conviene mostrar una sola persona y sus brazos/manos tanto en la foto como en el
video. La transferencia es generativa: manos, objetos y personajes 3D requieren
evaluación visual y no se garantiza una copia exacta. MuseTalk reutiliza su ciclo
de fotogramas cuando la voz dura más que la referencia.

La imagen Docker añade `/opt/motion-venv` aislado de MuseTalk y Wan. Se usa el
checkpoint original de 16 fotogramas, resolución corta de 320 píxeles y decodificación
por bloques de 2 para reducir memoria. Esto **no demuestra** que la ejecución
completa entre en 12 GB: falta medirla en la RTX 3060 del despliegue. DWPose usa
CPU; el generador usa CUDA. El primer trabajo descarga pesos oficiales de Tencent,
DWPose y SVD; necesita espacio libre y el acceso/licencia que requiera el proveedor.
`motion_installed` en `/health` comprueba instalación, no descarga ni inferencia.

Los errores de inferencia quedan en `/data/jobs/<id>-motion.log`. La API marca
el trabajo como fallido y limpia sus archivos temporales. El Studio descarga los
MP4 con autorización antes de reproducirlos y no reconstruye el reproductor en
cada consulta de estado sin cambios.

Validación sin GPU: `python -m unittest discover -s tests -v`. Estas pruebas
comprueban el orden movimiento→labios, el rechazo de fallos y límites de archivos;
no comprueban pesos, CUDA, calidad de movimiento ni sincronización real.
