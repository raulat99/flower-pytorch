"""pytorchexample: lógica del servidor federado para detección de matrículas."""

import json
import os

import boto3
import torch
from datetime import datetime, timezone
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pytorchexample.task import get_model, load_centralized_dataset, test

# Creamos la aplicación servidor de Flower
app = ServerApp()

s3 = boto3.client("s3")

BUCKET = os.environ["TFM_S3_BUCKET"]
RUN_ID = os.environ.get(
    "TFM_RUN_ID",
    datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-yolov8n-flower"),
)

LAST_SERVER_METRICS: dict = {}


def upload_json_to_s3(data: dict, key: str) -> None:
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=json.dumps(data, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def upload_file_to_s3(local_path: str, key: str) -> None:
    s3.upload_file(local_path, BUCKET, key)


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Punto de entrada principal del servidor. Orquesta todo el entrenamiento federado."""

    # Leemos la configuración del experimento definida en pyproject.toml
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Cargamos YOLOv8n con sus pesos iniciales (preentrenados en COCO)
    # Estos pesos se enviarán a todos los clientes en la primera ronda
    global_model = get_model()
    arrays = ArrayRecord(global_model.model.state_dict())

    # Usamos FedAvg como estrategia de agregación:
    # el servidor promedia los pesos de todos los clientes ponderando por número de ejemplos
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    # Arrancamos el bucle federado: train → aggregate → evaluate, durante num_rounds rondas
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),  # enviamos la lr a cada cliente
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,  # función para evaluar el modelo global en el servidor
    )

    # Al terminar guardamos el modelo global final en disco
    print("\nGuardando el modelo final en disco...")
    state_dict = result.arrays.to_torch_state_dict()
    model_path = "final_model.pt"
    torch.save(state_dict, model_path)

    model_key = f"runs/{RUN_ID}/models/final_model.pt"
    summary_key = f"runs/{RUN_ID}/metrics/summary.json"

    upload_file_to_s3(model_path, model_key)

    summary = {
        "run_id": RUN_ID,
        "status": "completed",
        "model": "yolov8n",
        "dataset": "keremberke/license-plate-object-detection",
        "model_s3_key": model_key,
        "final_metrics": LAST_SERVER_METRICS,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    upload_json_to_s3(summary, summary_key)


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evalúa el modelo global después de cada ronda usando el dataset de validación centralizado."""
    global LAST_SERVER_METRICS

    # Reconstruimos el modelo con los pesos agregados de esta ronda
    model = get_model()
    model.model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Cargamos el split de validación completo
    data_yaml = load_centralized_dataset()
    map50, map50_95, precision, recall = test(
        model, data_yaml, device
    )

    metrics = {
        "round": server_round,
        "map50": map50,
        "map50_95": map50_95,
        "precision": precision,
        "recall": recall,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    LAST_SERVER_METRICS = metrics

    upload_json_to_s3(
        metrics,
        f"runs/{RUN_ID}/metrics/round_{server_round:03d}.json",
    )

    # Devolvemos todas las métricas de detección para que Flower las registre en los logs
    return MetricRecord({
        "map50": map50,
        "map50_95": map50_95,
        "precision": precision,
        "recall": recall,
    })
