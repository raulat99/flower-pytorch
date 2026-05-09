"""pytorchexample: aplicación Flower / PyTorch para detección federada de matrículas."""

import copy
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from PIL import Image
from ultralytics import YOLO

# Nombre del dataset en HuggingFace y única clase que queremos detectar
DATASET_NAME = "keremberke/license-plate-object-detection"
CLASS_NAMES = ["license_plate"]

# Guardamos el split de entrenamiento en memoria para no descargarlo varias veces
FULL_TRAIN = None

# Evita imprimir las mismas estadísticas de dataset demasiadas veces
_PRINTED_DATASET_STATS: set[str] = set()


def get_model() -> YOLO:
    """Crea el modelo YOLO de una clase y carga pesos iniciales si están disponibles.

    El modelo se construye desde custom_yolov8.yaml para mantener nc=1.
    Después se intenta cargar yolo26n.pt como pesos iniciales compatibles.
    """
    _here = Path(__file__).resolve().parent
    model_yaml = _here / "custom_yolov8.yaml"
    weights = _here / "yolo26n.pt"

    model = YOLO(str(model_yaml))

    if weights.exists():
        try:
            model.load(str(weights))
            print(f"[MODEL] Pesos iniciales cargados desde: {weights}")
        except Exception as exc:
            print(
                "[MODEL] WARNING: no se pudieron cargar los pesos "
                f"desde {weights}. Se usará el modelo inicializado desde YAML.\n"
                f"[MODEL] Motivo: {exc}"
            )
    else:
        try:
            print("[MODEL] Descargando pesos yolov8n.pt desde ultralytics...")
            tmp = YOLO("yolov8n.pt")  # ultralytics lo descarga automáticamente
            model.load(str(tmp.ckpt_path))
            print("[MODEL] Pesos yolov8n.pt cargados correctamente.")
        except Exception as exc:
            print(
                f"[MODEL] WARNING: no se pudieron descargar pesos. "
                "Se usará el modelo inicializado desde YAML.\n"
                f"[MODEL] Motivo: {exc}"
            )

    return model


def _extract_bboxes(example: dict[str, Any]) -> list[list[float]]:
    """Extrae bounding boxes del ejemplo de HuggingFace.

    El dataset está en formato COCO: [x_min, y_min, width, height].
    Esta función es algo más robusta por si la estructura exacta cambia.
    """
    bboxes = []

    objects = example.get("objects", {})

    if isinstance(objects, dict):
        for key in ("bbox", "bboxes", "boxes"):
            if key in objects and objects[key] is not None:
                bboxes = objects[key]
                break

    elif isinstance(objects, list):
        extracted = []
        for obj in objects:
            if isinstance(obj, dict):
                bbox = obj.get("bbox") or obj.get("box") or obj.get("bboxes")
                if bbox is not None:
                    extracted.append(bbox)
        bboxes = extracted

    if not bboxes:
        for key in ("bbox", "bboxes", "boxes"):
            if key in example and example[key] is not None:
                bboxes = example[key]
                break

    clean_bboxes: list[list[float]] = []

    for bbox in bboxes:
        if bbox is None:
            continue

        if isinstance(bbox, dict):
            bbox = bbox.get("bbox") or bbox.get("box")

        try:
            coords = list(bbox)
        except TypeError:
            continue

        if len(coords) < 4:
            continue

        try:
            x_min = float(coords[0])
            y_min = float(coords[1])
            width = float(coords[2])
            height = float(coords[3])
        except (TypeError, ValueError):
            continue

        clean_bboxes.append([x_min, y_min, width, height])

    return clean_bboxes


def _clear_yolo_output_dir(output_dir: str) -> None:
    """Elimina un split YOLO temporal para regenerarlo limpio."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)


def _save_examples_to_yolo(examples, output_dir: str) -> None:
    """Convierte ejemplos COCO/HuggingFace a formato YOLO.

    YOLO espera una estructura como:

        output_dir/
            images/
                000000.jpg
            labels/
                000000.txt

    Cada línea del .txt tiene el formato:

        class x_center y_center width height

    con coordenadas normalizadas entre 0 y 1.
    """
    img_dir = os.path.join(output_dir, "images")
    lbl_dir = os.path.join(output_dir, "labels")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    total_images = 0
    total_boxes = 0
    images_without_boxes = 0

    for idx, example in enumerate(examples):
        img = example["image"]

        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        img = img.convert("RGB")
        img_w, img_h = img.size

        image_name = f"{idx:06d}.jpg"
        label_name = f"{idx:06d}.txt"

        img_path = os.path.join(img_dir, image_name)
        lbl_path = os.path.join(lbl_dir, label_name)

        img.save(img_path)

        bboxes = _extract_bboxes(example)

        valid_boxes_for_image = 0

        with open(lbl_path, "w", encoding="utf-8") as f:
            for bbox in bboxes:
                x_min, y_min, box_w, box_h = bbox

                # Ignoramos cajas inválidas
                if img_w <= 0 or img_h <= 0:
                    continue
                if box_w <= 0 or box_h <= 0:
                    continue

                # Conversión COCO -> YOLO
                x_center = (x_min + box_w / 2.0) / img_w
                y_center = (y_min + box_h / 2.0) / img_h
                box_w_norm = box_w / img_w
                box_h_norm = box_h / img_h

                # Clampeamos por seguridad al rango [0, 1]
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                box_w_norm = max(0.0, min(1.0, box_w_norm))
                box_h_norm = max(0.0, min(1.0, box_h_norm))

                # Si tras normalizar queda una caja degenerada, la ignoramos
                if box_w_norm <= 0.0 or box_h_norm <= 0.0:
                    continue

                # Clase 0 porque solo tenemos license_plate
                f.write(
                    f"0 {x_center:.6f} {y_center:.6f} "
                    f"{box_w_norm:.6f} {box_h_norm:.6f}\n"
                )

                valid_boxes_for_image += 1
                total_boxes += 1

        if valid_boxes_for_image == 0:
            images_without_boxes += 1

        total_images += 1

    print(f"[DATASET] Guardadas {total_images} imágenes en {output_dir}")
    print(f"[DATASET] Guardadas {total_boxes} bounding boxes en formato YOLO")
    print(f"[DATASET] Imágenes sin cajas: {images_without_boxes}")


def _count_files(folder: str, suffixes: tuple[str, ...]) -> int:
    if not os.path.exists(folder):
        return 0

    count = 0
    for filename in os.listdir(folder):
        if filename.lower().endswith(suffixes):
            count += 1

    return count


def _dataset_stats(output_dir: str) -> dict[str, int]:
    """Devuelve estadísticas básicas de un split YOLO."""
    img_dir = os.path.join(output_dir, "images")
    lbl_dir = os.path.join(output_dir, "labels")

    num_images = _count_files(img_dir, (".jpg", ".jpeg", ".png"))
    num_labels = _count_files(lbl_dir, (".txt",))

    non_empty_labels = 0
    total_boxes = 0

    if os.path.exists(lbl_dir):
        for filename in os.listdir(lbl_dir):
            if not filename.lower().endswith(".txt"):
                continue

            path = os.path.join(lbl_dir, filename)

            if os.path.getsize(path) > 0:
                non_empty_labels += 1

            with open(path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                total_boxes += len(lines)

    return {
        "images": num_images,
        "labels": num_labels,
        "non_empty_labels": non_empty_labels,
        "boxes": total_boxes,
    }


def _has_valid_yolo_dataset(output_dir: str) -> bool:
    """Comprueba que el split YOLO tiene imágenes y etiquetas útiles."""
    stats = _dataset_stats(output_dir)

    if stats["images"] <= 0:
        return False

    # Queremos un .txt por imagen. Aunque alguna imagen no tenga caja,
    # debe existir su .txt vacío para mantener el dataset consistente.
    if stats["labels"] != stats["images"]:
        return False

    # Para este dataset de matrículas, debe haber al menos una caja.
    # Si boxes=0, Ultralytics mostrará "no labels found".
    if stats["boxes"] <= 0:
        return False

    return True


def _print_dataset_stats_once(split_name: str, output_dir: str) -> None:
    key = f"{split_name}:{output_dir}"

    if key in _PRINTED_DATASET_STATS:
        return

    stats = _dataset_stats(output_dir)

    print(
        f"[DATASET] {split_name}: "
        f"{stats['images']} imágenes, "
        f"{stats['labels']} ficheros .txt, "
        f"{stats['non_empty_labels']} labels no vacías, "
        f"{stats['boxes']} cajas"
    )

    _PRINTED_DATASET_STATS.add(key)


def _ensure_yolo_dataset(examples, output_dir: str, split_name: str) -> None:
    """Asegura que el split existe y contiene labels válidas.

    Si existe images/ pero no labels/, o si las labels están vacías,
    se borra y se regenera el split entero.
    """
    if not _has_valid_yolo_dataset(output_dir):
        print(f"[DATASET] Regenerando split '{split_name}' en: {output_dir}")
        _clear_yolo_output_dir(output_dir)
        _save_examples_to_yolo(examples, output_dir)

    stats = _dataset_stats(output_dir)

    print(
        f"[DATASET] {split_name}: "
        f"{stats['images']} imágenes, "
        f"{stats['labels']} ficheros .txt, "
        f"{stats['non_empty_labels']} labels no vacías, "
        f"{stats['boxes']} cajas"
    )

    if stats["images"] <= 0:
        raise RuntimeError(
            f"No se han generado imágenes para el split '{split_name}'. "
            f"Ruta: {output_dir}"
        )

    if stats["labels"] != stats["images"]:
        raise RuntimeError(
            f"Dataset YOLO inconsistente en '{split_name}': "
            f"{stats['images']} imágenes pero {stats['labels']} labels. "
            f"Ruta: {output_dir}"
        )

    if stats["boxes"] <= 0:
        raise RuntimeError(
            f"No se han generado bounding boxes válidas para '{split_name}'. "
            "Ultralytics no podrá calcular mAP. "
            "Revisa la extracción de 'objects/bbox' del dataset."
        )


def _write_data_yaml(base_dir: str, train_rel: str, val_rel: str) -> str:
    """Genera el fichero data.yaml que necesita YOLO."""
    yaml_path = os.path.join(base_dir, "data.yaml")

    # En Windows es más seguro escribir rutas con /
    base_path = Path(base_dir).resolve().as_posix()

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {base_path}\n")
        f.write(f"train: {train_rel}\n")
        f.write(f"val: {val_rel}\n")
        f.write("nc: 1\n")
        f.write(f"names: {CLASS_NAMES}\n")

    print(f"[DATASET] data.yaml generado en: {yaml_path}")
    return yaml_path


def load_data(partition_id: int, num_partitions: int) -> tuple[str, int, int]:
    """Carga y prepara la partición de datos correspondiente a un cliente federado.

    Devuelve:
        - ruta al data.yaml
        - número de ejemplos de entrenamiento
        - número de ejemplos de validación local
    """
    global FULL_TRAIN

    if FULL_TRAIN is None:
        print("[DATASET] Cargando split train desde HuggingFace...")
        FULL_TRAIN = load_dataset(
            DATASET_NAME,
            name="full",
            split="train",
            trust_remote_code=True,
        ).shuffle(seed=42)

    n = len(FULL_TRAIN)

    start = (partition_id * n) // num_partitions
    end = ((partition_id + 1) * n) // num_partitions

    partition = FULL_TRAIN.select(range(start, end))
    splits = partition.train_test_split(test_size=0.2, seed=42)

    base_dir = os.path.join(
        tempfile.gettempdir(),
        f"lp_yolo_partition_{partition_id}",
    )

    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")

    _ensure_yolo_dataset(
        splits["train"],
        train_dir,
        f"cliente_{partition_id}_train",
    )

    _ensure_yolo_dataset(
        splits["test"],
        val_dir,
        f"cliente_{partition_id}_val",
    )

    yaml_path = _write_data_yaml(base_dir, "train/images", "val/images")

    return yaml_path, len(splits["train"]), len(splits["test"])


def load_centralized_dataset() -> str:
    """Prepara el split de validación centralizado para evaluar el modelo global."""
    base_dir = os.path.join(tempfile.gettempdir(), "lp_yolo_central")
    val_dir = os.path.join(base_dir, "validation")

    if not _has_valid_yolo_dataset(val_dir):
        print("[DATASET] Cargando split validation desde HuggingFace...")
        valid_data = load_dataset(
            DATASET_NAME,
            name="full",
            split="validation",
            trust_remote_code=True,
        )

        _ensure_yolo_dataset(
            valid_data,
            val_dir,
            "servidor_validation",
        )
    else:
        _print_dataset_stats_once("servidor_validation", val_dir)

    return _write_data_yaml(base_dir, "validation/images", "validation/images")


def _yolo_device(device: torch.device) -> int | str:
    """Convierte el device PyTorch al formato que entiende Ultralytics."""
    device_str = str(device)

    if device_str.startswith("cuda"):
        return int(device_str.split(":")[-1]) if ":" in device_str else 0

    return "cpu"


def _yolo_with_weights(net: YOLO) -> YOLO:
    """Recarga el modelo desde un .pt temporal para que Ultralytics preserve los pesos.

    Si YOLO se inicializa desde YAML, algunas versiones de Ultralytics pueden
    reinicializar durante .train(). Guardar el modelo actual como .pt temporal
    y recargarlo evita que se pierdan los pesos globales recibidos desde Flower.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    tmp.close()

    try:
        model_copy = copy.deepcopy(net.model).cpu().float()

        torch.save(
            {
                "model": model_copy,
                "epoch": 0,
                "train_args": {},
            },
            tmp.name,
        )

        return YOLO(tmp.name)

    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def _get_float_from_dict(data: dict[str, Any], keys: list[str], default: float = 0.0) -> float:
    """Obtiene un float de un diccionario probando varias claves."""
    for key in keys:
        value = data.get(key)

        if value is None:
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return default


def train(
    net: YOLO,
    data_yaml: str,
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[YOLO, float]:
    """Entrena el modelo YOLO con los datos locales del cliente."""
    base_dir = os.path.dirname(data_yaml)

    # Recargamos desde .pt para que YOLO no descarte los pesos federados
    net = _yolo_with_weights(net)

    results = net.train(
        data=data_yaml,
        epochs=epochs,
        lr0=lr,
        device=_yolo_device(device),
        verbose=False,
        plots=False,
        workers=0,
        cache=False,
        project=os.path.join(base_dir, "runs"),
        name="train",
        exist_ok=True,
    )

    train_loss = 0.0

    try:
        results_dict = getattr(results, "results_dict", {}) or {}

        # Algunas versiones de Ultralytics no guardan estas pérdidas en results_dict.
        # Si no existen, devolvemos 0.0 solo como métrica auxiliar de Flower.
        box_loss = _get_float_from_dict(
            results_dict,
            ["train/box_loss", "box_loss", "metrics/box_loss"],
            0.0,
        )

        cls_loss = _get_float_from_dict(
            results_dict,
            ["train/cls_loss", "cls_loss", "metrics/cls_loss"],
            0.0,
        )

        dfl_loss = _get_float_from_dict(
            results_dict,
            ["train/dfl_loss", "dfl_loss", "metrics/dfl_loss"],
            0.0,
        )

        train_loss = box_loss + cls_loss + dfl_loss

    except Exception as exc:
        print(f"[TRAIN] WARNING: no se pudo leer train_loss: {exc}")
        train_loss = 0.0

    return net, train_loss


def test(net: YOLO, data_yaml: str, device: torch.device) -> tuple[float, float]:
    """Evalúa el modelo y devuelve loss aproximada y mAP50."""
    base_dir = os.path.dirname(data_yaml)
    run_name = f"val_{Path(base_dir).name}"

    metrics = net.val(
        data=data_yaml,
        device=_yolo_device(device),
        verbose=False,
        plots=False,
        workers=0,
        project=os.path.join(tempfile.gettempdir(), "lp_yolo_val_runs"),
        name=run_name,
        exist_ok=True,
    )

    try:
        results_dict = getattr(metrics, "results_dict", {}) or {}

        val_loss = _get_float_from_dict(
            results_dict,
            [
                "val/box_loss",
                "val/cls_loss",
                "val/dfl_loss",
                "metrics/box_loss",
            ],
            0.0,
        )

        # La forma más estable suele ser metrics.box.map50.
        map50 = float(metrics.box.map50)

    except Exception as exc:
        print(f"[EVAL] WARNING: no se pudieron leer métricas de validación: {exc}")
        val_loss = 0.0
        map50 = 0.0

    print(f"[EVAL] data={data_yaml} | loss={val_loss:.6f} | map50={map50:.6f}")

    return val_loss, map50