"""pytorchexample: aplicación Flower / PyTorch para detección de matrículas."""

import copy
import os
import tempfile

import torch
from datasets import load_dataset
from PIL import Image
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from pathlib import Path

# Nombre del dataset en HuggingFace y la única clase que queremos detectar
DATASET_NAME = "keremberke/license-plate-object-detection"
CLASS_NAMES = ["license_plate"]

# Guardamos el split de entrenamiento en memoria para no descargarlo varias veces
_full_train = None


def get_model() -> YOLO:
    model_yaml = Path(__file__).resolve().parents[1] / "custom_yolov8.yaml"
    return YOLO(str(model_yaml))
    #base = YOLO("custom_yolov8.yaml")
    #return base

def _save_examples_to_yolo(examples, output_dir: str) -> None:
    """Convierte los ejemplos del dataset al formato que espera YOLO y los guarda en disco.

    El dataset viene en formato COCO: las bounding boxes son [x_min, y_min, w, h]
    en píxeles absolutos. YOLO necesita [clase x_centro y_centro ancho alto]
    con valores normalizados entre 0 y 1 respecto al tamaño de la imagen.
    """
    # Creamos las carpetas de imágenes y etiquetas si no existen
    img_dir = os.path.join(output_dir, "images")
    lbl_dir = os.path.join(output_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for idx, example in enumerate(examples):
        img = example["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")
        w, h = img.size

        # Guardamos la imagen en disco
        img.save(os.path.join(img_dir, f"{idx:06d}.jpg"))

        # Escribimos el archivo de etiquetas con las bounding boxes convertidas
        with open(os.path.join(lbl_dir, f"{idx:06d}.txt"), "w") as f:
            objects = example["objects"]
            for bbox in objects.get("bbox", []):
                x_min, y_min, bw, bh = bbox
                # Conversión de COCO a YOLO: centro normalizado y tamaño normalizado
                x_c = (x_min + bw / 2) / w
                y_c = (y_min + bh / 2) / h
                # Clase 0 porque solo tenemos una clase (license_plate)
                f.write(f"0 {x_c:.6f} {y_c:.6f} {bw / w:.6f} {bh / h:.6f}\n")


def _write_data_yaml(base_dir: str, train_rel: str, val_rel: str) -> str:
    """Genera el fichero data.yaml que necesita YOLO para encontrar los datos."""
    yaml_path = os.path.join(base_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {base_dir}\n")   # carpeta raíz del dataset
        f.write(f"train: {train_rel}\n")  # ruta relativa a las imágenes de entrenamiento
        f.write(f"val: {val_rel}\n")      # ruta relativa a las imágenes de validación
        f.write("nc: 1\n")               # número de clases
        f.write(f"names: {CLASS_NAMES}\n") # nombres de las clases
    return yaml_path


def load_data(partition_id: int, num_partitions: int) -> tuple[str, int, int]:
    """Carga y prepara la partición de datos correspondiente a este cliente federado.

    Devuelve la ruta al data.yaml, el número de ejemplos de entrenamiento y de validación.
    """
    global _full_train
    # Solo descargamos el dataset completo la primera vez
    if _full_train is None:
        _full_train = load_dataset(
            DATASET_NAME, name="full", split="train", trust_remote_code=True
        ).shuffle(seed=42)  # barajamos para que la partición sea aleatoria pero reproducible

    # Dividimos el dataset en franjas iguales, una por cliente (partición IID)
    n = len(_full_train)
    start = (partition_id * n) // num_partitions
    end = ((partition_id + 1) * n) // num_partitions
    partition = _full_train.select(range(start, end))

    # Dentro de la partición de cada cliente, separamos 80% train y 20% validación local
    splits = partition.train_test_split(test_size=0.2, seed=42)

    # Usamos una carpeta temporal distinta para cada cliente
    base_dir = os.path.join(
        tempfile.gettempdir(), f"lp_yolo_partition_{partition_id}"
    )
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")

    # Solo escribimos en disco si no lo hemos hecho antes (para ahorrar tiempo)
    if not os.path.exists(os.path.join(train_dir, "images")):
        _save_examples_to_yolo(splits["train"], train_dir)
    if not os.path.exists(os.path.join(val_dir, "images")):
        _save_examples_to_yolo(splits["test"], val_dir)

    yaml_path = _write_data_yaml(base_dir, "train/images", "val/images")
    return yaml_path, len(splits["train"]), len(splits["test"])


def load_centralized_dataset() -> str:
    """Prepara el split de validación completo para que el servidor pueda evaluar el modelo global."""
    base_dir = os.path.join(tempfile.gettempdir(), "lp_yolo_central")
    val_dir = os.path.join(base_dir, "validation")

    # Solo lo escribimos en disco la primera vez
    if not os.path.exists(os.path.join(val_dir, "images")):
        valid_data = load_dataset(DATASET_NAME, name="full", split="validation", trust_remote_code=True)
        _save_examples_to_yolo(valid_data, val_dir)

    return _write_data_yaml(base_dir, "validation/images", "validation/images")


def _yolo_device(device: torch.device) -> int | str:
    """Convierte el device de PyTorch al formato que entiende ultralytics (número de GPU o 'cpu')."""
    s = str(device)
    if s.startswith("cuda"):
        return int(s.split(":")[-1]) if ":" in s else 0
    return "cpu"


def _yolo_with_weights(net: YOLO) -> YOLO:
    """Guarda los pesos actuales en un .pt temporal y recarga YOLO desde él.

    YOLO solo preserva los pesos cargados durante .train() si el modelo fue
    inicializado desde un checkpoint .pt (self.ckpt != None). Si se cargó
    desde un YAML, reinicializa desde cero ignorando cualquier load_state_dict().
    Guardarlo en .pt temporal y recargarlo fuerza self.ckpt != None.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    tmp.close()
    try:
        torch.save({"model": copy.deepcopy(net.model).half(), "epoch": 0}, tmp.name)
        return YOLO(tmp.name)
    finally:
        os.unlink(tmp.name)


def train(net: YOLO, data_yaml: str, epochs: int, lr: float, device: torch.device) -> tuple["YOLO", float]:
    """Entrena el modelo YOLOv8n con los datos locales del cliente durante un número de épocas.

    Devuelve el modelo entrenado y la pérdida de las bounding boxes al final del entrenamiento.
    """
    base_dir = os.path.dirname(data_yaml)
    # Recargamos desde .pt para que YOLO no descarte los pesos del servidor al entrenar
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
    try:
        # La clave correcta en ultralytics es "train/box_loss" o "metrics/box_loss"
        # dependiendo de la versión; probamos ambas
        rd = results.results_dict
        box_loss = float(
            rd.get("train/box_loss") or rd.get("metrics/box_loss") or 0.0
        )
    except (AttributeError, TypeError):
        box_loss = 0.0
    return net, box_loss


def test(net: YOLO, data_yaml: str, device: torch.device) -> tuple[float, float]:
    """Evalúa el modelo sobre un conjunto de datos y devuelve la pérdida y el mAP50.

    El mAP50 (mean Average Precision al 50% de IoU) es la métrica estándar
    para medir la calidad de un detector de objetos.
    """
    metrics = net.val(
        data=data_yaml,
        device=_yolo_device(device),
        verbose=False,
        plots=False,
    )
    # Recogemos las métricas (si algo falla devolvemos ceros para no romper nada)
    try:
        val_loss = float(metrics.results_dict.get("val/box_loss", 0.0))
        map50 = float(metrics.box.map50)
    except (AttributeError, TypeError):
        val_loss, map50 = 0.0, 0.0
    return val_loss, map50
