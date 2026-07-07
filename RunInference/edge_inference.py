import json
import os
import re
import shutil
import tempfile
import time
import base64
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import torch
from PIL import Image
from ultralytics import YOLO

try:
    from fast_plate_ocr import LicensePlateRecognizer
except ImportError as exc:
    raise RuntimeError(
        "Falta 'fast-plate-ocr'. Instálalo con:\n"
        "  pip install 'fast-plate-ocr[onnx]'\n"
        "En Raspberry Pi, si tu imagen no trae wheels para alguna dependencia, prueba primero:\n"
        "  pip install --upgrade pip setuptools wheel\n"
        "  pip install 'fast-plate-ocr[onnx]'"
    ) from exc


# ==========================================================
# Configuración general
# ==========================================================
BUCKET = os.environ.get("TFM_S3_BUCKET")
DEVICE_ID = os.environ.get("DEVICE_ID", "raspberry-unknown")
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH")
MODEL_RUN_ID_ENV = os.environ.get("MODEL_RUN_ID", "local-model")

# ==========================================================
# Configuración API Gateway + Cognito
# ==========================================================
# Si estas variables están definidas, el script usa la API protegida por Cognito.
# Si no lo están, mantiene el modo local/S3 como fallback para pruebas.
API_BASE_URL = os.environ.get("API_BASE_URL", "").rstrip("/")
COGNITO_TOKEN_URL = os.environ.get("COGNITO_TOKEN_URL", "")
COGNITO_CLIENT_ID = os.environ.get("COGNITO_CLIENT_ID", "")
COGNITO_CLIENT_SECRET = os.environ.get("COGNITO_CLIENT_SECRET", "")
COGNITO_SCOPES = os.environ.get(
    "COGNITO_SCOPES",
    "tfm-api/read-watchlist "
    "tfm-api/write-inference "
    "tfm-api/write-alert "
    "tfm-api/read-model "
    "tfm-api/device-heartbeat",
)
API_TIMEOUT_SECONDS = float(os.environ.get("API_TIMEOUT_SECONDS", "15"))
API_ENABLED = bool(
    API_BASE_URL
    and COGNITO_TOKEN_URL
    and COGNITO_CLIENT_ID
    and COGNITO_CLIENT_SECRET
)
SEND_SUMMARIES_TO_API = os.environ.get("SEND_SUMMARIES_TO_API", "1") == "1"
SEND_ALERTS_TO_API = os.environ.get("SEND_ALERTS_TO_API", "1") == "1"
LOCAL_WATCHLIST_FALLBACK = os.environ.get("LOCAL_WATCHLIST_FALLBACK", "1") == "1"

# Descarga de modelo por API y heartbeat del dispositivo.
# MODEL_API_FALLBACK_TO_S3=1 permite volver al método antiguo sólo si falla /models/latest.
MODEL_API_FALLBACK_TO_S3 = os.environ.get("MODEL_API_FALLBACK_TO_S3", "0") == "1"
HEARTBEAT_INTERVAL_SECONDS = float(os.environ.get("HEARTBEAT_INTERVAL_SECONDS", "60"))
SOFTWARE_VERSION = os.environ.get("SOFTWARE_VERSION", "1.0.0")

_TOKEN_CACHE: dict[str, Any] = {
    "access_token": None,
    "expires_at": 0.0,
}

LATEST_MODEL_KEY = os.environ.get(
    "LATEST_MODEL_KEY",
    "models/production/latest_model.json",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
INPUT_DIR = BASE_DIR / "input_images"
PROCESSING_DIR = BASE_DIR / "processing_images"
PROCESSED_DIR = BASE_DIR / "processed_images"
FAILED_DIR = BASE_DIR / "failed_images"
OUTPUT_DIR = BASE_DIR / "output"
SUMMARY_DIR = OUTPUT_DIR / "summaries"
ALERTS_DIR = OUTPUT_DIR / "alerts"
WATCHLIST_PATH = Path(os.environ.get("WATCHLIST_PATH", str(BASE_DIR / "watchlist.json")))

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

# Polling simple: el script revisa input_images/ cada X segundos.
POLL_INTERVAL_SECONDS = float(os.environ.get("POLL_INTERVAL_SECONDS", "1.0"))
FILE_STABILITY_CHECKS = int(os.environ.get("FILE_STABILITY_CHECKS", "3"))
FILE_STABILITY_DELAY_SECONDS = float(os.environ.get("FILE_STABILITY_DELAY_SECONDS", "0.4"))
WATCHLIST_REFRESH_SECONDS = float(os.environ.get("WATCHLIST_REFRESH_SECONDS", "60"))

# Envío de resumen agregado cada N imágenes procesadas correctamente.
SUMMARY_BATCH_SIZE = int(os.environ.get("SUMMARY_BATCH_SIZE", "25"))

# Configuración de detección y OCR.
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.25"))
YOLO_DEVICE = os.environ.get("YOLO_DEVICE", "cpu")
OCR_MODEL_NAME = os.environ.get("OCR_MODEL_NAME", "cct-s-v2-global-model")
OCR_CROP_PADDING = float(os.environ.get("OCR_CROP_PADDING", "0.08"))
OCR_MIN_CROP_WIDTH = int(os.environ.get("OCR_MIN_CROP_WIDTH", "240"))

# Privacidad: por defecto NO se guardan crops de todas las matrículas.
# Sólo se guardan crops/evidencias cuando hay coincidencia con watchlist.
SAVE_MATCH_PLATE_CROPS = os.environ.get("SAVE_MATCH_PLATE_CROPS", "1") == "1"
# Si API está configurada, por privacidad preferimos API Gateway/Lambda en vez de subida directa a S3.
UPLOAD_SUMMARIES_TO_S3 = os.environ.get("UPLOAD_SUMMARIES_TO_S3", "0" if API_ENABLED else "1") == "1"
UPLOAD_ALERTS_TO_S3 = os.environ.get("UPLOAD_ALERTS_TO_S3", "0" if API_ENABLED else "1") == "1"
SUMMARY_S3_PREFIX = os.environ.get("SUMMARY_S3_PREFIX", "inference_summaries")
ALERTS_S3_PREFIX = os.environ.get("ALERTS_S3_PREFIX", "alerts")

# Validación genérica de matrículas europeas. No se filtra por un único país.
EUROPE_GENERIC_MIN_LEN = int(os.environ.get("EUROPE_GENERIC_MIN_LEN", "3"))
EUROPE_GENERIC_MAX_LEN = int(os.environ.get("EUROPE_GENERIC_MAX_LEN", "12"))

for directory in (
    MODEL_DIR,
    INPUT_DIR,
    PROCESSING_DIR,
    PROCESSED_DIR,
    FAILED_DIR,
    OUTPUT_DIR,
    SUMMARY_DIR,
    ALERTS_DIR,
):
    directory.mkdir(exist_ok=True)

s3 = boto3.client("s3") if BUCKET else None


COMMON_EU_PLATE_PATTERNS: dict[str, list[str]] = {
    "ES_modern": [r"^\d{4}[BCDFGHJKLMNPRSTVWXYZ]{3}$"],
    "FR_modern": [r"^[A-Z]{2}\d{3}[A-Z]{2}$"],
    "IT_modern": [r"^[A-Z]{2}\d{3}[A-Z]{2}$"],
    "PT_common": [r"^[A-Z]{2}\d{2}[A-Z]{2}$", r"^\d{2}[A-Z]{2}\d{2}$", r"^\d{4}[A-Z]{2}$"],
    "UK_current": [r"^[A-Z]{2}\d{2}[A-Z]{3}$"],
    "DE_common": [r"^[A-Z]{1,3}[A-Z]{1,2}\d{1,4}$"],
    "NL_common": [r"^[A-Z]{2}\d{2}[A-Z]{2}$", r"^\d{2}[A-Z]{2}\d{2}$", r"^[A-Z]{2}\d{4}$"],
    "BE_common": [r"^\d[A-Z]{3}\d{3}$", r"^[A-Z]{3}\d{3}$"],
}


# ==========================================================
# Utilidades generales
# ==========================================================
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat()


def safe_timestamp(dt: datetime | None = None) -> str:
    dt = dt or utc_now()
    return dt.strftime("%Y%m%d-%H%M%S")


def safe_path_token(value: str) -> str:
    value = normalize_european_plate_text(value) or "unknown"
    return re.sub(r"[^A-Z0-9_-]", "_", value)


def write_json_atomic(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    tmp_path.replace(path)


def unique_destination(directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / filename

    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    ts = safe_timestamp()

    for idx in range(1, 1000):
        candidate = directory / f"{stem}_{ts}_{idx:03d}{suffix}"
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"No se pudo generar un nombre único para {filename} en {directory}")


def move_file_to_directory(path: Path, directory: Path) -> Path:
    destination = unique_destination(directory, path.name)
    shutil.move(str(path), str(destination))
    return destination


def upload_file_to_s3(local_path: Path, key: str) -> bool:
    if not BUCKET or s3 is None:
        print(f"[{DEVICE_ID}] S3 no configurado. No se sube {local_path}")
        return False

    s3.upload_file(str(local_path), BUCKET, key)
    print(f"[{DEVICE_ID}] Subido a s3://{BUCKET}/{key}")
    return True


# ==========================================================
# Cliente API Gateway + Cognito
# ==========================================================
def _read_http_error(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")
    except Exception:
        return str(exc)


def get_access_token() -> str:
    """Obtiene y cachea un access token M2M de Cognito."""
    if not API_ENABLED:
        raise RuntimeError(
            "API no configurada. Define API_BASE_URL, COGNITO_TOKEN_URL, "
            "COGNITO_CLIENT_ID y COGNITO_CLIENT_SECRET."
        )

    now = time.time()
    cached_token = _TOKEN_CACHE.get("access_token")
    expires_at = float(_TOKEN_CACHE.get("expires_at") or 0.0)

    if cached_token and now < expires_at - 60:
        return str(cached_token)

    form_data = urllib.parse.urlencode(
        {
            "grant_type": "client_credentials",
            "scope": COGNITO_SCOPES,
        }
    ).encode("utf-8")

    basic = base64.b64encode(
        f"{COGNITO_CLIENT_ID}:{COGNITO_CLIENT_SECRET}".encode("utf-8")
    ).decode("ascii")

    request = urllib.request.Request(
        COGNITO_TOKEN_URL,
        data=form_data,
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {basic}",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Error obteniendo token Cognito: {exc.code} {_read_http_error(exc)}") from exc

    access_token = payload["access_token"]
    expires_in = int(payload.get("expires_in", 3600))

    _TOKEN_CACHE["access_token"] = access_token
    _TOKEN_CACHE["expires_at"] = now + expires_in

    return access_token


def api_request(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Llama a API Gateway usando Authorization: Bearer <token>."""
    if not path.startswith("/"):
        path = "/" + path

    body = None
    headers = {
        "Authorization": f"Bearer {get_access_token()}",
        "Accept": "application/json",
    }

    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    if extra_headers:
        headers.update(extra_headers)

    request = urllib.request.Request(
        API_BASE_URL + path,
        data=body,
        method=method,
        headers=headers,
    )

    try:
        with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Error API {method} {path}: {exc.code} {_read_http_error(exc)}"
        ) from exc


def _parse_watchlist_items(items: Any) -> dict[str, dict[str, Any]]:
    watchlist: dict[str, dict[str, Any]] = {}

    if not isinstance(items, list):
        return watchlist

    for item in items:
        if isinstance(item, str):
            plate_raw = item
            metadata: dict[str, Any] = {
                "plate_text": item,
                "status": "active",
            }
        elif isinstance(item, dict):
            plate_raw = (
                item.get("plate_text")
                or item.get("plate")
                or item.get("license_plate")
                or item.get("matricula")
            )
            metadata = dict(item)
        else:
            continue

        plate_text = normalize_european_plate_text(str(plate_raw or ""))
        if not plate_text:
            continue

        status = str(metadata.get("status", "active")).lower()
        if status not in {"active", "activo", "enabled", "true"}:
            continue

        metadata["plate_text"] = plate_text
        watchlist[plate_text] = metadata

    return watchlist


def fetch_watchlist_from_api() -> dict[str, dict[str, Any]]:
    data = api_request("GET", "/watchlist/active")
    items = data.get("plates") or data.get("watchlist") or data.get("items") or []
    watchlist = _parse_watchlist_items(items)
    print(f"[{DEVICE_ID}] Watchlist API cargada: {len(watchlist)} matrículas activas")
    return watchlist


def post_inference_summary_to_api(summary: dict[str, Any]) -> None:
    response = api_request("POST", "/inference-runs", summary)
    print(
        f"[{DEVICE_ID}] Summary enviado a API: "
        f"{response.get('run_id') or response.get('message') or 'OK'}"
    )


def _alert_payload_for_api(event: dict[str, Any]) -> dict[str, Any]:
    """Evita enviar rutas locales internas a DynamoDB/API."""
    payload = dict(event)
    payload.pop("local_crop_path", None)
    payload.pop("event_s3_key", None)
    payload["api_reported_at"] = iso_now()
    return payload


def post_watchlist_match_to_api(event: dict[str, Any]) -> str:
    response = api_request("POST", "/watchlist/matches", _alert_payload_for_api(event))
    alert_id = str(response.get("alert_id") or event["alert_id"])
    print(f"[{DEVICE_ID}] Alerta enviada a API: {alert_id}")
    return alert_id


def get_evidence_upload_url_from_api(
    alert_id: str,
    device_id: str,
    plate_text: str,
    filename: str,
    content_type: str = "image/jpeg",
) -> dict[str, Any]:
    return api_request(
        "POST",
        f"/alerts/{urllib.parse.quote(alert_id, safe='')}/evidence-url",
        {
            "device_id": device_id,
            "plate_text": plate_text,
            "filename": filename,
            "content_type": content_type,
        },
    )


def upload_file_to_presigned_url(upload_url: str, local_path: Path, content_type: str = "image/jpeg") -> None:
    with open(local_path, "rb") as f:
        data = f.read()

    request = urllib.request.Request(
        upload_url,
        data=data,
        method="PUT",
        headers={"Content-Type": content_type},
    )

    try:
        with urllib.request.urlopen(request, timeout=max(API_TIMEOUT_SECONDS, 30)) as resp:
            if resp.status not in (200, 201, 204):
                raise RuntimeError(f"S3 presigned PUT devolvió status {resp.status}")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Error subiendo evidencia a URL prefirmada: {exc.code} {_read_http_error(exc)}") from exc

    print(f"[{DEVICE_ID}] Evidencia subida mediante URL prefirmada: {local_path.name}")


def download_file_from_url(url: str, destination: Path) -> None:
    """Descarga un fichero desde una URL prefirmada de forma atómica."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")

    request = urllib.request.Request(url, method="GET")

    try:
        with urllib.request.urlopen(request, timeout=max(API_TIMEOUT_SECONDS, 60)) as resp:
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(resp, f)
    except urllib.error.HTTPError as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f"Error descargando modelo desde URL prefirmada: {exc.code} {_read_http_error(exc)}"
        ) from exc
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    tmp_path.replace(destination)


def fetch_latest_model_from_api() -> dict[str, Any]:
    """Consulta GET /models/latest y devuelve metadata + URL prefirmada de descarga."""
    data = api_request("GET", "/models/latest")

    if not data.get("model_download_url"):
        raise RuntimeError("La respuesta de /models/latest no contiene model_download_url")

    if not data.get("model_s3_key"):
        raise RuntimeError("La respuesta de /models/latest no contiene model_s3_key")

    return data


def post_device_heartbeat_to_api(model_metadata: dict[str, Any]) -> None:
    """Notifica al backend que la Raspberry sigue activa."""
    if not API_ENABLED:
        return

    payload = {
        "device_id": DEVICE_ID,
        "status": "online",
        "current_model_run_id": model_metadata.get("run_id"),
        "model_run_id": model_metadata.get("run_id"),
        "software_version": SOFTWARE_VERSION,
    }

    response = api_request("POST", "/devices/heartbeat", payload)
    print(
        f"[{DEVICE_ID}] Heartbeat enviado a API: "
        f"{response.get('message') or 'OK'}"
    )


# ==========================================================
# Modelo y OCR
# ==========================================================
def download_latest_model() -> tuple[Path, dict[str, Any]]:
    """Descarga el último modelo aprobado.

    Prioridad:
      1. LOCAL_MODEL_PATH para pruebas locales.
      2. API Gateway: GET /models/latest + URL prefirmada.
      3. Fallback legado S3 directo sólo si API no está configurada o MODEL_API_FALLBACK_TO_S3=1.
    """
    if LOCAL_MODEL_PATH:
        model_path = Path(LOCAL_MODEL_PATH).expanduser().resolve()
        if not model_path.exists():
            raise RuntimeError(f"LOCAL_MODEL_PATH no existe: {model_path}")

        metadata = {
            "run_id": MODEL_RUN_ID_ENV,
            "model_s3_key": None,
            "source": "local",
            "local_model_path": str(model_path),
        }
        print(f"[{DEVICE_ID}] Usando modelo local: {model_path}")
        return model_path, metadata

    local_model_path = MODEL_DIR / "final_model.pt"
    metadata_path = MODEL_DIR / "model_metadata.json"

    previous: dict[str, Any] = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            previous = json.load(f)

    if API_ENABLED:
        try:
            print(f"[{DEVICE_ID}] Consultando último modelo por API: GET /models/latest")
            latest = fetch_latest_model_from_api()

            model_key = latest["model_s3_key"]
            run_id = latest.get("run_id", "unknown-run")
            previous_key = previous.get("model_s3_key")

            metadata = {
                **latest,
                "run_id": run_id,
                "model_s3_key": model_key,
                "source": "api",
                "api_endpoint": "/models/latest",
                "downloaded_at": previous.get("downloaded_at"),
            }

            # No guardamos la URL prefirmada en metadata permanente porque caduca.
            model_download_url = metadata.pop("model_download_url", latest["model_download_url"])

            if local_model_path.exists() and previous_key == model_key:
                print(f"[{DEVICE_ID}] Modelo ya descargado y coincide con API: {local_model_path}")
                metadata.update(previous)
                metadata["run_id"] = previous.get("run_id", run_id)
                metadata["model_s3_key"] = model_key
                metadata["source"] = "api"
                return local_model_path, metadata

            print(f"[{DEVICE_ID}] Descargando modelo aprobado mediante URL prefirmada")
            download_file_from_url(model_download_url, local_model_path)

            metadata["downloaded_at"] = iso_now()
            metadata["presigned_url_expires_in"] = latest.get("expires_in")
            write_json_atomic(metadata, metadata_path)

            return local_model_path, metadata

        except Exception as exc:
            if not MODEL_API_FALLBACK_TO_S3:
                raise RuntimeError(
                    "Error descargando el modelo desde la API. "
                    "Si quieres permitir fallback directo a S3, define MODEL_API_FALLBACK_TO_S3=1. "
                    f"Detalle: {exc}"
                ) from exc

            print(f"[{DEVICE_ID}] ERROR modelo por API: {exc}")
            print(f"[{DEVICE_ID}] MODEL_API_FALLBACK_TO_S3=1, se intenta método S3 directo")

    if not BUCKET or s3 is None:
        raise RuntimeError(
            "No se ha definido TFM_S3_BUCKET ni LOCAL_MODEL_PATH. "
            "Define API_BASE_URL/COGNITO_* para descargar por API, "
            "TFM_S3_BUCKET para fallback S3, o LOCAL_MODEL_PATH para probar en local."
        )

    latest_path = MODEL_DIR / "latest_model.json"

    print(f"[{DEVICE_ID}] Descargando puntero de modelo por S3 directo: s3://{BUCKET}/{LATEST_MODEL_KEY}")
    s3.download_file(BUCKET, LATEST_MODEL_KEY, str(latest_path))

    with open(latest_path, "r", encoding="utf-8") as f:
        latest = json.load(f)

    model_key = latest["model_s3_key"]
    run_id = latest.get("run_id", "unknown-run")
    previous_key = previous.get("model_s3_key")

    metadata = {
        "run_id": run_id,
        "model_s3_key": model_key,
        "latest_model_key": LATEST_MODEL_KEY,
        "source": "s3_direct",
    }

    if local_model_path.exists() and previous_key == model_key:
        print(f"[{DEVICE_ID}] Modelo ya descargado: {local_model_path}")
        metadata.update(previous)
        metadata["run_id"] = previous.get("run_id", run_id)
        metadata["model_s3_key"] = model_key
        return local_model_path, metadata

    print(f"[{DEVICE_ID}] Descargando modelo aprobado por S3 directo: s3://{BUCKET}/{model_key}")
    s3.download_file(BUCKET, model_key, str(local_model_path))

    metadata["downloaded_at"] = iso_now()
    write_json_atomic(metadata, metadata_path)

    return local_model_path, metadata

def load_flower_yolo_model(model_path: Path) -> YOLO:
    """Carga el modelo federado guardado como state_dict sobre custom_yolov8.yaml."""
    model_yaml = BASE_DIR / "custom_yolov8.yaml"

    if not model_yaml.exists():
        raise RuntimeError(
            f"No existe {model_yaml}. Copia custom_yolov8.yaml junto a este script."
        )

    model = YOLO(str(model_yaml))

    state_dict = torch.load(
        str(model_path),
        map_location="cpu",
    )

    model.model.load_state_dict(state_dict)
    model.model.names = {0: "license_plate"}
    model.model.nc = 1
    model.model.eval()

    return model


def load_plate_ocr() -> Any:
    """Inicializa fast-plate-ocr una única vez para reutilizarlo."""
    print(f"[{DEVICE_ID}] Cargando OCR fast-plate-ocr: {OCR_MODEL_NAME}")
    ocr = LicensePlateRecognizer(OCR_MODEL_NAME)
    print(f"[{DEVICE_ID}] OCR fast-plate-ocr cargado correctamente")
    return ocr


# ==========================================================
# OCR y normalización de matrículas
# ==========================================================
def _clip_int(value: float, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, int(round(value))))


def crop_plate_from_bbox(
    image: Image.Image,
    bbox_xyxy: dict[str, float],
    padding_ratio: float = OCR_CROP_PADDING,
) -> tuple[Image.Image, dict[str, int]]:
    """Recorta la matrícula detectada, añadiendo un pequeño margen."""
    img_w, img_h = image.size

    x1 = float(bbox_xyxy["x1"])
    y1 = float(bbox_xyxy["y1"])
    x2 = float(bbox_xyxy["x2"])
    y2 = float(bbox_xyxy["y2"])

    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)

    pad_x = box_w * padding_ratio
    pad_y = box_h * padding_ratio

    crop_x1 = _clip_int(x1 - pad_x, 0, img_w)
    crop_y1 = _clip_int(y1 - pad_y, 0, img_h)
    crop_x2 = _clip_int(x2 + pad_x, 0, img_w)
    crop_y2 = _clip_int(y2 + pad_y, 0, img_h)

    if crop_x2 <= crop_x1:
        crop_x2 = min(img_w, crop_x1 + 1)
    if crop_y2 <= crop_y1:
        crop_y2 = min(img_h, crop_y1 + 1)

    crop_bbox = {
        "x1": crop_x1,
        "y1": crop_y1,
        "x2": crop_x2,
        "y2": crop_y2,
    }

    return image.crop((crop_x1, crop_y1, crop_x2, crop_y2)), crop_bbox


def prepare_plate_crop(crop: Image.Image) -> Image.Image:
    """Prepara el recorte para fast-plate-ocr sin binarizarlo ni pasarlo a gris."""
    img = crop.convert("RGB")
    width, height = img.size

    if width < OCR_MIN_CROP_WIDTH:
        scale = OCR_MIN_CROP_WIDTH / max(1, width)
        new_size = (int(width * scale), int(height * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    return img


def normalize_european_plate_text(text: str) -> str:
    """Normaliza matrícula: mayúsculas y sólo A-Z/0-9."""
    text = text.upper().strip()
    return re.sub(r"[^A-Z0-9]", "", text)


def is_plausible_european_plate(text: str) -> bool:
    """Validación amplia para no descartar matrículas europeas de distintos formatos."""
    if not re.fullmatch(r"[A-Z0-9]+", text or ""):
        return False

    if not (EUROPE_GENERIC_MIN_LEN <= len(text) <= EUROPE_GENERIC_MAX_LEN):
        return False

    has_letter = any(ch.isalpha() for ch in text)
    has_digit = any(ch.isdigit() for ch in text)
    return has_letter and has_digit


def match_common_eu_plate_formats(text: str) -> list[str]:
    """Devuelve candidatos de formato europeo común, sin imponer un único país."""
    matches: list[str] = []

    for country_or_format, patterns in COMMON_EU_PLATE_PATTERNS.items():
        if any(re.fullmatch(pattern, text) for pattern in patterns):
            matches.append(country_or_format)

    return matches


def _as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        values = [_as_float_or_none(v) for v in value]
        clean = [v for v in values if v is not None]
        return sum(clean) / len(clean) if clean else None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_prediction_field(prediction: Any, field_names: tuple[str, ...]) -> Any:
    if isinstance(prediction, dict):
        for field in field_names:
            if field in prediction:
                return prediction[field]
        return None

    for field in field_names:
        if hasattr(prediction, field):
            return getattr(prediction, field)

    return None


def parse_fast_plate_prediction(prediction: Any) -> dict[str, Any]:
    """Convierte la salida de fast-plate-ocr a un diccionario estable."""
    if isinstance(prediction, str):
        raw_text = prediction
    else:
        raw_text = _get_prediction_field(
            prediction,
            (
                "plate",
                "text",
                "prediction",
                "license_plate",
                "plate_text",
                "value",
            ),
        )
        if raw_text is None:
            raw_text = str(prediction)

    raw_text = str(raw_text or "").strip()
    plate_text = normalize_european_plate_text(raw_text)

    confidence = _as_float_or_none(
        _get_prediction_field(
            prediction,
            (
                "confidence",
                "prob",
                "score",
                "plate_confidence",
                "plate_prob",
                "char_prob",
                "char_probs",
            ),
        )
    )

    region = _get_prediction_field(prediction, ("region", "country", "country_code"))
    region_prob = _as_float_or_none(
        _get_prediction_field(prediction, ("region_prob", "region_confidence"))
    )

    return {
        "plate_text": plate_text,
        "plate_text_raw": raw_text,
        "ocr_confidence": confidence,
        "ocr_region": str(region) if region is not None else None,
        "ocr_region_confidence": region_prob,
        "ocr_engine": "fast-plate-ocr",
        "ocr_model": OCR_MODEL_NAME,
        "plate_format_profile": "europe_generic",
        "is_plausible_european_plate": is_plausible_european_plate(plate_text),
        "europe_country_format_candidates": match_common_eu_plate_formats(plate_text),
    }


def read_plate_text(ocr: Any, crop: Image.Image) -> dict[str, Any]:
    """Lee el texto de una matrícula. Usa fichero temporal y no guarda crops generales."""
    processed = prepare_plate_crop(crop)
    temp_path: Path | None = None

    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        temp_path = Path(tmp.name)
        processed.save(temp_path, quality=95)

        try:
            predictions = ocr.run(str(temp_path), return_confidence=True)
        except TypeError:
            predictions = ocr.run(str(temp_path))

        if not isinstance(predictions, (list, tuple)):
            predictions = [predictions]

        if not predictions:
            return parse_fast_plate_prediction("")

        return parse_fast_plate_prediction(predictions[0])

    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


# ==========================================================
# Watchlist API/local
# ==========================================================
def load_watchlist(path: Path = WATCHLIST_PATH) -> dict[str, dict[str, Any]]:
    """
    Carga la watchlist desde API Gateway + Cognito si está configurado.

    Si la API no está configurada o falla y LOCAL_WATCHLIST_FALLBACK=1,
    usa watchlist.json local para pruebas offline.
    """
    if API_ENABLED:
        try:
            return fetch_watchlist_from_api()
        except Exception as exc:
            if not LOCAL_WATCHLIST_FALLBACK:
                raise
            print(f"[{DEVICE_ID}] ERROR cargando watchlist desde API: {exc}")
            print(f"[{DEVICE_ID}] Se intenta fallback local: {path}")

    if not path.exists():
        print(f"[{DEVICE_ID}] Watchlist local no encontrada en {path}. Se usará lista vacía.")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("plates") or data.get("watchlist") or data.get("items") or []
    else:
        raise RuntimeError(f"Formato de watchlist no soportado: {type(data)}")

    watchlist = _parse_watchlist_items(items)
    print(f"[{DEVICE_ID}] Watchlist local cargada: {len(watchlist)} matrículas activas")
    return watchlist


# ==========================================================
# Polling de imágenes
# ==========================================================
def is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and not path.name.startswith(".")


def wait_until_file_is_stable(path: Path) -> bool:
    """Evita leer una imagen mientras aún se está copiando/escribiendo."""
    previous_size = -1

    for _ in range(FILE_STABILITY_CHECKS):
        if not path.exists():
            return False

        current_size = path.stat().st_size
        if current_size > 0 and current_size == previous_size:
            return True

        previous_size = current_size
        time.sleep(FILE_STABILITY_DELAY_SECONDS)

    return False


def get_next_pending_image() -> Path | None:
    candidates = sorted(path for path in INPUT_DIR.iterdir() if is_supported_image(path))

    for path in candidates:
        if wait_until_file_is_stable(path):
            return path

    return None


# ==========================================================
# Resúmenes agregados y alertas
# ==========================================================
def create_summary_window(model_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "device_id": DEVICE_ID,
        "window_start": iso_now(),
        "summary_batch_size": SUMMARY_BATCH_SIZE,
        "privacy_mode": "aggregate_only_except_watchlist_matches",
        "model_run_id": model_metadata.get("run_id"),
        "model_s3_key": model_metadata.get("model_s3_key"),
        "ocr_engine": "fast-plate-ocr",
        "ocr_model": OCR_MODEL_NAME,
        "plate_format_profile": "europe_generic",
        "num_images": 0,
        "num_total_detections": 0,
        "num_ocr_attempts": 0,
        "num_ocr_success": 0,
        "num_plausible_european_plates": 0,
        "num_watchlist_matches": 0,
        "num_failed_images": 0,
    }


def merge_image_stats(summary: dict[str, Any], image_stats: dict[str, int]) -> None:
    for key in (
        "num_images",
        "num_total_detections",
        "num_ocr_attempts",
        "num_ocr_success",
        "num_plausible_european_plates",
        "num_watchlist_matches",
        "num_failed_images",
    ):
        summary[key] = int(summary.get(key, 0)) + int(image_stats.get(key, 0))


def send_summary(summary: dict[str, Any], force: bool = False) -> None:
    """Guarda el resumen agregado localmente y lo envía a la API cada 25 imágenes."""
    if not force and int(summary.get("num_images", 0)) < SUMMARY_BATCH_SIZE:
        return

    if int(summary.get("num_images", 0)) == 0 and int(summary.get("num_failed_images", 0)) == 0:
        return

    now = utc_now()
    ts = safe_timestamp(now)

    payload = dict(summary)
    payload["window_end"] = now.isoformat()
    payload["timestamp"] = now.isoformat()
    payload["api_enabled"] = API_ENABLED

    local_path = SUMMARY_DIR / f"summary_{ts}.json"
    write_json_atomic(payload, local_path)
    print(
        f"[{DEVICE_ID}] Summary generado: {local_path} "
        f"({payload['num_images']} imágenes, {payload['num_watchlist_matches']} matches)"
    )

    if API_ENABLED and SEND_SUMMARIES_TO_API:
        try:
            post_inference_summary_to_api(payload)
        except Exception as exc:
            print(f"[{DEVICE_ID}] ERROR enviando summary a API: {exc}")

    # Fallback legado opcional. Por defecto queda desactivado cuando API_ENABLED=True.
    if UPLOAD_SUMMARIES_TO_S3:
        key = f"{SUMMARY_S3_PREFIX}/{DEVICE_ID}/{ts}/summary.json"
        upload_file_to_s3(local_path, key)


def save_alert_crop(crop: Image.Image, alert_dir: Path, image_path: Path, detection_index: int) -> Path | None:
    if not SAVE_MATCH_PLATE_CROPS:
        return None

    crop_name = f"{image_path.stem}_plate_{detection_index:03d}.jpg"
    crop_path = alert_dir / crop_name
    alert_dir.mkdir(parents=True, exist_ok=True)

    processed = prepare_plate_crop(crop)
    processed.save(crop_path, quality=95)
    return crop_path


def create_alert_event(
    image_path: Path,
    detection_index: int,
    bbox_xyxy: dict[str, float],
    ocr_crop_bbox_xyxy: dict[str, int],
    yolo_confidence: float,
    class_id: int,
    ocr_result: dict[str, Any],
    watchlist_item: dict[str, Any],
    crop: Image.Image,
    model_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Crea y reporta evidencia sólo para matrículas que coinciden con la watchlist."""
    now = utc_now()
    ts = safe_timestamp(now)
    plate_text = str(ocr_result.get("plate_text") or "UNKNOWN")
    plate_token = safe_path_token(plate_text)
    alert_id = f"alert-{ts}-{DEVICE_ID}-{plate_token}-{detection_index:03d}"

    alert_dir = ALERTS_DIR / ts / plate_token
    crop_path = save_alert_crop(crop, alert_dir, image_path, detection_index)

    # Con API Gateway, la evidencia se sube con URL prefirmada generada por Lambda.
    # Calculamos la misma clave que generará la Lambda para poder guardarla en DynamoDB
    # desde el POST /watchlist/matches.
    crop_s3_key = None
    if API_ENABLED and crop_path is not None:
        crop_s3_key = f"alerts/{DEVICE_ID}/{alert_id}/{plate_text}/{crop_path.name}"
    elif UPLOAD_ALERTS_TO_S3 and crop_path is not None:
        crop_s3_key = f"{ALERTS_S3_PREFIX}/{DEVICE_ID}/{ts}/{plate_token}/{crop_path.name}"
        upload_file_to_s3(crop_path, crop_s3_key)

    event_s3_key = None
    if not API_ENABLED and UPLOAD_ALERTS_TO_S3:
        event_s3_key = f"{ALERTS_S3_PREFIX}/{DEVICE_ID}/{ts}/{plate_token}/event.json"

    event = {
        "alert_id": alert_id,
        "device_id": DEVICE_ID,
        "timestamp": now.isoformat(),
        "plate_text": plate_text,
        "watchlist_item": watchlist_item,
        "image_name": image_path.name,
        "detection_index": detection_index,
        "class_id": class_id,
        "class_name": "license_plate",
        "confidence": yolo_confidence,
        "yolo_confidence": yolo_confidence,
        "bbox_xyxy": bbox_xyxy,
        "ocr_crop_bbox_xyxy": ocr_crop_bbox_xyxy,
        "ocr_confidence": ocr_result.get("ocr_confidence"),
        "ocr_region": ocr_result.get("ocr_region"),
        "ocr_region_confidence": ocr_result.get("ocr_region_confidence"),
        "ocr_engine": ocr_result.get("ocr_engine"),
        "ocr_model": ocr_result.get("ocr_model"),
        "plate_format_profile": ocr_result.get("plate_format_profile"),
        "is_plausible_european_plate": ocr_result.get("is_plausible_european_plate"),
        "europe_country_format_candidates": ocr_result.get("europe_country_format_candidates", []),
        "model_run_id": model_metadata.get("run_id"),
        "model_s3_key": model_metadata.get("model_s3_key"),
        "local_crop_path": str(crop_path) if crop_path else None,
        "crop_s3_key": crop_s3_key,
        "event_s3_key": event_s3_key,
        "privacy": {
            "full_results_json_uploaded": False,
            "full_image_uploaded": False,
            "only_watchlist_match_uploaded": True,
            "plate_crop_uploaded": False,
            "upload_mechanism": "api_presigned_url" if API_ENABLED else "direct_s3_or_local",
        },
        "api": {
            "enabled": API_ENABLED,
            "match_posted": False,
            "evidence_upload_requested": False,
            "evidence_uploaded": False,
        },
    }

    event_path = alert_dir / "event.json"
    write_json_atomic(event, event_path)

    if API_ENABLED and SEND_ALERTS_TO_API:
        try:
            api_alert_id = post_watchlist_match_to_api(event)
            event["api"]["match_posted"] = True
            event["api"]["alert_id_returned"] = api_alert_id

            if crop_path is not None:
                evidence = get_evidence_upload_url_from_api(
                    alert_id=alert_id,
                    device_id=DEVICE_ID,
                    plate_text=plate_text,
                    filename=crop_path.name,
                    content_type="image/jpeg",
                )
                event["api"]["evidence_upload_requested"] = True
                event["api"]["evidence_s3_key_returned"] = evidence.get("s3_key")

                upload_file_to_presigned_url(
                    upload_url=evidence["upload_url"],
                    local_path=crop_path,
                    content_type="image/jpeg",
                )
                event["api"]["evidence_uploaded"] = True
                event["privacy"]["plate_crop_uploaded"] = True

        except Exception as exc:
            event["api"]["error"] = repr(exc)
            print(f"[{DEVICE_ID}] ERROR enviando alerta/evidencia a API: {exc}")

        write_json_atomic(event, event_path)

    elif UPLOAD_ALERTS_TO_S3 and event_s3_key:
        upload_file_to_s3(event_path, event_s3_key)
        event["privacy"]["plate_crop_uploaded"] = bool(crop_s3_key)
        write_json_atomic(event, event_path)

    print(f"[{DEVICE_ID}] ALERTA watchlist: {plate_text} ({alert_id})")
    return event


# ==========================================================
# Inferencia de una imagen
# ==========================================================
def process_one_image(
    model: YOLO,
    ocr: Any,
    image_path: Path,
    watchlist: dict[str, dict[str, Any]],
    model_metadata: dict[str, Any],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Procesa una imagen y devuelve sólo estadísticas agregadas y matches."""
    print(f"[{DEVICE_ID}] Inferencia sobre: {image_path.name}")

    image_stats = {
        "num_images": 1,
        "num_total_detections": 0,
        "num_ocr_attempts": 0,
        "num_ocr_success": 0,
        "num_plausible_european_plates": 0,
        "num_watchlist_matches": 0,
        "num_failed_images": 0,
    }
    matches: list[dict[str, Any]] = []
    matched_plates_in_image: set[str] = set()

    with Image.open(image_path) as img:
        original_image = img.convert("RGB")

    results = model.predict(
        source=str(image_path),
        conf=YOLO_CONF,
        save=False,  # privacidad: no guardar imágenes anotadas de todas las detecciones
        verbose=False,
        device=YOLO_DEVICE,
    )

    detection_index = 0

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            detection_index += 1
            image_stats["num_total_detections"] += 1
            image_stats["num_ocr_attempts"] += 1

            xyxy = box.xyxy[0].tolist()
            yolo_confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            bbox_xyxy = {
                "x1": xyxy[0],
                "y1": xyxy[1],
                "x2": xyxy[2],
                "y2": xyxy[3],
            }

            plate_crop, ocr_crop_bbox = crop_plate_from_bbox(original_image, bbox_xyxy)
            ocr_result = read_plate_text(ocr, plate_crop)
            plate_text = str(ocr_result.get("plate_text") or "")

            if plate_text:
                image_stats["num_ocr_success"] += 1

            if ocr_result.get("is_plausible_european_plate"):
                image_stats["num_plausible_european_plates"] += 1

            watchlist_item = watchlist.get(plate_text)
            if watchlist_item is None:
                continue

            # Evita duplicar la misma matrícula varias veces en una misma imagen.
            if plate_text in matched_plates_in_image:
                continue

            matched_plates_in_image.add(plate_text)
            image_stats["num_watchlist_matches"] += 1

            alert_event = create_alert_event(
                image_path=image_path,
                detection_index=detection_index,
                bbox_xyxy=bbox_xyxy,
                ocr_crop_bbox_xyxy=ocr_crop_bbox,
                yolo_confidence=yolo_confidence,
                class_id=class_id,
                ocr_result=ocr_result,
                watchlist_item=watchlist_item,
                crop=plate_crop,
                model_metadata=model_metadata,
            )
            matches.append(alert_event)

    return image_stats, matches


# ==========================================================
# Servicio continuo
# ==========================================================
def run_daemon() -> None:
    model_path, model_metadata = download_latest_model()

    print(f"[{DEVICE_ID}] Cargando modelo YOLO: {model_path}")
    model = load_flower_yolo_model(model_path)
    ocr = load_plate_ocr()

    watchlist = load_watchlist()
    last_watchlist_refresh = time.monotonic()
    last_heartbeat = time.monotonic()

    if API_ENABLED:
        try:
            post_device_heartbeat_to_api(model_metadata)
        except Exception as exc:
            print(f"[{DEVICE_ID}] ERROR enviando heartbeat inicial: {exc}")

    summary_window = create_summary_window(model_metadata)

    print(
        f"[{DEVICE_ID}] Servicio de inferencia continua iniciado. "
        f"Vigilando carpeta: {INPUT_DIR}"
    )
    print(f"[{DEVICE_ID}] Summary agregado cada {SUMMARY_BATCH_SIZE} imágenes procesadas")
    print(f"[{DEVICE_ID}] API Gateway/Cognito: {'activado' if API_ENABLED else 'desactivado'}")
    if API_ENABLED:
        print(f"[{DEVICE_ID}] API_BASE_URL={API_BASE_URL}")

    try:
        while True:
            if API_ENABLED and time.monotonic() - last_heartbeat >= HEARTBEAT_INTERVAL_SECONDS:
                try:
                    post_device_heartbeat_to_api(model_metadata)
                except Exception as exc:
                    print(f"[{DEVICE_ID}] ERROR enviando heartbeat: {exc}")
                last_heartbeat = time.monotonic()

            # Recarga periódica de la watchlist local.
            if time.monotonic() - last_watchlist_refresh >= WATCHLIST_REFRESH_SECONDS:
                try:
                    watchlist = load_watchlist()
                except Exception as exc:
                    print(f"[{DEVICE_ID}] ERROR recargando watchlist: {exc}")
                last_watchlist_refresh = time.monotonic()

            pending_image = get_next_pending_image()
            if pending_image is None:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            processing_path = move_file_to_directory(pending_image, PROCESSING_DIR)

            try:
                image_stats, _matches = process_one_image(
                    model=model,
                    ocr=ocr,
                    image_path=processing_path,
                    watchlist=watchlist,
                    model_metadata=model_metadata,
                )
                merge_image_stats(summary_window, image_stats)

                processed_path = move_file_to_directory(processing_path, PROCESSED_DIR)
                print(f"[{DEVICE_ID}] Imagen procesada movida a: {processed_path}")

                if int(summary_window.get("num_images", 0)) >= SUMMARY_BATCH_SIZE:
                    send_summary(summary_window, force=True)
                    summary_window = create_summary_window(model_metadata)

            except KeyboardInterrupt:
                raise

            except Exception as exc:
                error_ts = safe_timestamp()
                error_payload = {
                    "device_id": DEVICE_ID,
                    "timestamp": iso_now(),
                    "image_name": processing_path.name,
                    "error": repr(exc),
                }
                error_path = FAILED_DIR / f"{processing_path.stem}_{error_ts}_error.json"
                write_json_atomic(error_payload, error_path)
                failed_path = move_file_to_directory(processing_path, FAILED_DIR)

                merge_image_stats(summary_window, {"num_failed_images": 1})
                print(f"[{DEVICE_ID}] ERROR procesando imagen. Movida a {failed_path}. Motivo: {exc}")

    except KeyboardInterrupt:
        print(f"\n[{DEVICE_ID}] Parando servicio por KeyboardInterrupt...")

    finally:
        # No se pierden métricas pendientes si se detiene el proceso antes de llegar a 25 imágenes.
        send_summary(summary_window, force=True)
        print(f"[{DEVICE_ID}] Servicio detenido")


if __name__ == "__main__":
    run_daemon()
