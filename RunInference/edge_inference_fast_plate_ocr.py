import json
import os
import re
import tempfile
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


BUCKET = os.environ["TFM_S3_BUCKET"]
DEVICE_ID = os.environ.get("DEVICE_ID", "raspberry-unknown")

LATEST_MODEL_KEY = "models/production/latest_model.json"

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
INPUT_DIR = BASE_DIR / "input_images"
OUTPUT_DIR = BASE_DIR / "output"
PLATE_CROPS_DIR = OUTPUT_DIR / "plate_crops"

# Configuración de detección y OCR. Se puede sobrescribir con variables de entorno.
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.25"))
OCR_MODEL_NAME = os.environ.get("OCR_MODEL_NAME", "cct-s-v2-global-model")
OCR_CROP_PADDING = float(os.environ.get("OCR_CROP_PADDING", "0.08"))
OCR_MIN_CROP_WIDTH = int(os.environ.get("OCR_MIN_CROP_WIDTH", "240"))
SAVE_PLATE_CROPS = os.environ.get("SAVE_PLATE_CROPS", "1") == "1"

# No se valida contra un único país porque el objetivo es soportar matrículas europeas variadas.
# Se guarda una validación genérica y, si encaja, candidatos de país/formato comunes.
EUROPE_GENERIC_MIN_LEN = int(os.environ.get("EUROPE_GENERIC_MIN_LEN", "3"))
EUROPE_GENERIC_MAX_LEN = int(os.environ.get("EUROPE_GENERIC_MAX_LEN", "12"))

MODEL_DIR.mkdir(exist_ok=True)
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
PLATE_CROPS_DIR.mkdir(exist_ok=True)

s3 = boto3.client("s3")


COMMON_EU_PLATE_PATTERNS: dict[str, list[str]] = {
    # Patrones frecuentes tras eliminar espacios, guiones y símbolos.
    # No son exhaustivos: sirven como pista, no como filtro duro.
    "ES_modern": [r"^\d{4}[BCDFGHJKLMNPRSTVWXYZ]{3}$"],
    "FR_modern": [r"^[A-Z]{2}\d{3}[A-Z]{2}$"],
    "IT_modern": [r"^[A-Z]{2}\d{3}[A-Z]{2}$"],
    "PT_common": [r"^[A-Z]{2}\d{2}[A-Z]{2}$", r"^\d{2}[A-Z]{2}\d{2}$", r"^\d{4}[A-Z]{2}$"],
    "UK_current": [r"^[A-Z]{2}\d{2}[A-Z]{3}$"],
    "DE_common": [r"^[A-Z]{1,3}[A-Z]{1,2}\d{1,4}$"],
    "NL_common": [r"^[A-Z]{2}\d{2}[A-Z]{2}$", r"^\d{2}[A-Z]{2}\d{2}$", r"^[A-Z]{2}\d{4}$"],
    "BE_common": [r"^\d[A-Z]{3}\d{3}$", r"^[A-Z]{3}\d{3}$"],
}


def download_latest_model() -> Path:
    latest_path = MODEL_DIR / "latest_model.json"

    print(f"[{DEVICE_ID}] Descargando puntero de modelo: s3://{BUCKET}/{LATEST_MODEL_KEY}")

    s3.download_file(
        BUCKET,
        LATEST_MODEL_KEY,
        str(latest_path),
    )

    with open(latest_path, "r", encoding="utf-8") as f:
        latest = json.load(f)

    model_key = latest["model_s3_key"]
    run_id = latest.get("run_id", "unknown-run")

    local_model_path = MODEL_DIR / "final_model.pt"
    metadata_path = MODEL_DIR / "model_metadata.json"

    previous_key = None
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            previous = json.load(f)
            previous_key = previous.get("model_s3_key")

    if local_model_path.exists() and previous_key == model_key:
        print(f"[{DEVICE_ID}] Modelo ya descargado: {local_model_path}")
        return local_model_path

    print(f"[{DEVICE_ID}] Descargando modelo aprobado: s3://{BUCKET}/{model_key}")

    s3.download_file(
        BUCKET,
        model_key,
        str(local_model_path),
    )

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "model_s3_key": model_key,
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )

    return local_model_path


def load_flower_yolo_model(model_path: Path) -> YOLO:
    """Carga el modelo federado guardado como state_dict sobre tu custom_yolov8.yaml."""
    model_yaml = BASE_DIR / "custom_yolov8.yaml"

    if not model_yaml.exists():
        raise RuntimeError(
            f"No existe {model_yaml}. Copia custom_yolov8.yaml junto a edge_inference.py"
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
    """Inicializa fast-plate-ocr una única vez para reutilizarlo en todos los recortes."""
    print(f"[{DEVICE_ID}] Cargando OCR fast-plate-ocr: {OCR_MODEL_NAME}")
    ocr = LicensePlateRecognizer(OCR_MODEL_NAME)
    print(f"[{DEVICE_ID}] OCR fast-plate-ocr cargado correctamente")
    return ocr


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

    # Si el recorte llega muy pequeño, lo ampliamos antes de guardarlo/pasarlo al OCR.
    # Mantener RGB suele funcionar mejor que binarizar para modelos entrenados con deep learning.
    if width < OCR_MIN_CROP_WIDTH:
        scale = OCR_MIN_CROP_WIDTH / max(1, width)
        new_size = (int(width * scale), int(height * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    return img


def normalize_european_plate_text(text: str) -> str:
    """Normaliza matrícula europea genérica: mayúsculas y solo A-Z/0-9."""
    text = text.upper().strip()
    return re.sub(r"[^A-Z0-9]", "", text)


def is_plausible_european_plate(text: str) -> bool:
    """Validación amplia para no descartar matrículas europeas de distintos formatos."""
    if not re.fullmatch(r"[A-Z0-9]+", text or ""):
        return False

    if not (EUROPE_GENERIC_MIN_LEN <= len(text) <= EUROPE_GENERIC_MAX_LEN):
        return False

    # La mayoría de matrículas europeas ordinarias mezclan letras y números.
    # No lo usamos como filtro duro por país, pero sí para marcar plausibilidad.
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
    """Convierte la salida de fast-plate-ocr a un diccionario estable para el JSON."""
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
    region_prob = _as_float_or_none(_get_prediction_field(prediction, ("region_prob", "region_confidence")))

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


def save_plate_crop(
    crop: Image.Image,
    image_path: Path,
    detection_index: int,
) -> tuple[str | None, Path | None]:
    """Guarda el recorte de matrícula para poder auditar el OCR."""
    if not SAVE_PLATE_CROPS:
        return None, None

    crop_name = f"{image_path.stem}_plate_{detection_index:03d}.jpg"
    crop_path = PLATE_CROPS_DIR / crop_name
    crop.convert("RGB").save(crop_path, quality=95)

    return str(crop_path.relative_to(OUTPUT_DIR)), crop_path


def read_plate_text(ocr: Any, crop: Image.Image, image_path: Path, detection_index: int) -> dict[str, Any]:
    """Lee el texto de una matrícula usando fast-plate-ocr."""
    processed = prepare_plate_crop(crop)
    crop_relative_path, crop_path = save_plate_crop(processed, image_path, detection_index)

    temp_path: Path | None = None

    try:
        # La API documentada usa ruta de imagen. Si no guardamos crops, usamos un temporal.
        if crop_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.close()
            temp_path = Path(tmp.name)
            processed.save(temp_path, quality=95)
            ocr_input_path = temp_path
        else:
            ocr_input_path = crop_path

        try:
            predictions = ocr.run(str(ocr_input_path), return_confidence=True)
        except TypeError:
            # Compatibilidad con versiones antiguas sin return_confidence.
            predictions = ocr.run(str(ocr_input_path))

        if not isinstance(predictions, (list, tuple)):
            predictions = [predictions]

        if not predictions:
            result = parse_fast_plate_prediction("")
        else:
            result = parse_fast_plate_prediction(predictions[0])

        if crop_relative_path is not None:
            result["plate_crop"] = crop_relative_path

        return result

    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def choose_best_detection(detections: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Elige la mejor matrícula combinando confidence de YOLO y confidence del OCR."""
    if not detections:
        return None

    def score(det: dict[str, Any]) -> float:
        yolo_conf = float(det.get("confidence") or 0.0)
        ocr_conf = det.get("ocr_confidence")
        plausibility_boost = 1.10 if det.get("is_plausible_european_plate") else 1.0

        if ocr_conf is None:
            return yolo_conf * plausibility_boost

        return yolo_conf * max(0.0, float(ocr_conf)) * plausibility_boost

    best = max(detections, key=score)
    return {
        "plate_text": best.get("plate_text"),
        "confidence": best.get("confidence"),
        "ocr_confidence": best.get("ocr_confidence"),
        "ocr_region": best.get("ocr_region"),
        "ocr_region_confidence": best.get("ocr_region_confidence"),
        "is_plausible_european_plate": best.get("is_plausible_european_plate"),
        "europe_country_format_candidates": best.get("europe_country_format_candidates", []),
        "bbox_xyxy": best.get("bbox_xyxy"),
        "plate_crop": best.get("plate_crop"),
    }


def run_inference(model_path: Path) -> dict:
    print(f"[{DEVICE_ID}] Cargando modelo: {model_path}")
    model = load_flower_yolo_model(model_path)
    ocr = load_plate_ocr()

    image_paths: list[Path] = []
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(INPUT_DIR.glob(suffix))

    image_paths = sorted(image_paths)

    if not image_paths:
        raise RuntimeError(f"No hay imágenes en {INPUT_DIR}")

    all_results: list[dict[str, Any]] = []

    for image_path in image_paths:
        print(f"[{DEVICE_ID}] Inferencia sobre: {image_path.name}")

        original_image = Image.open(image_path).convert("RGB")

        results = model.predict(
            source=str(image_path),
            conf=YOLO_CONF,
            save=True,
            project=str(OUTPUT_DIR),
            name="predictions",
            exist_ok=True,
            verbose=False,
        )

        image_result: dict[str, Any] = {
            "image": image_path.name,
            "plate_texts": [],
            "best_plate": None,
            "detections": [],
        }

        detection_index = 0

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                detection_index += 1

                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                bbox_xyxy = {
                    "x1": xyxy[0],
                    "y1": xyxy[1],
                    "x2": xyxy[2],
                    "y2": xyxy[3],
                }

                plate_crop, ocr_crop_bbox = crop_plate_from_bbox(
                    original_image,
                    bbox_xyxy,
                )

                ocr_result = read_plate_text(
                    ocr=ocr,
                    crop=plate_crop,
                    image_path=image_path,
                    detection_index=detection_index,
                )

                detection = {
                    "class_id": cls,
                    "class_name": "license_plate",
                    "confidence": conf,
                    "bbox_xyxy": bbox_xyxy,
                    "ocr_crop_bbox_xyxy": ocr_crop_bbox,
                    **ocr_result,
                }

                image_result["detections"].append(detection)

                if ocr_result.get("plate_text"):
                    image_result["plate_texts"].append(ocr_result["plate_text"])

        # Evitamos duplicados conservando el orden.
        image_result["plate_texts"] = list(dict.fromkeys(image_result["plate_texts"]))
        image_result["best_plate"] = choose_best_detection(image_result["detections"])

        all_results.append(image_result)

    num_total_detections = sum(len(r["detections"]) for r in all_results)
    num_ocr_success = sum(
        1
        for r in all_results
        for d in r["detections"]
        if d.get("plate_text")
    )
    num_plausible_european = sum(
        1
        for r in all_results
        for d in r["detections"]
        if d.get("is_plausible_european_plate")
    )

    summary = {
        "device_id": DEVICE_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_images": len(image_paths),
        "num_total_detections": num_total_detections,
        "num_ocr_attempts": num_total_detections,
        "num_ocr_success": num_ocr_success,
        "num_plausible_european_plates": num_plausible_european,
        "ocr_engine": "fast-plate-ocr",
        "ocr_model": OCR_MODEL_NAME,
        "plate_format_profile": "europe_generic",
        "results": all_results,
    }

    return summary


def upload_results(summary: dict) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    key = f"inference/{DEVICE_ID}/{timestamp}/results.json"

    local_results_path = OUTPUT_DIR / "results.json"

    with open(local_results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    s3.upload_file(
        str(local_results_path),
        BUCKET,
        key,
    )

    print(f"[{DEVICE_ID}] Resultados subidos a s3://{BUCKET}/{key}")


def main():
    model_path = download_latest_model()
    summary = run_inference(model_path)
    upload_results(summary)

    print(f"[{DEVICE_ID}] Inferencia terminada")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
