"""pytorchexample: lógica del servidor federado para detección de matrículas."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pytorchexample.task import get_model, load_centralized_dataset, test

# Creamos la aplicación servidor de Flower
app = ServerApp()


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
    torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evalúa el modelo global después de cada ronda usando el dataset de validación centralizado."""

    # Reconstruimos el modelo con los pesos agregados de esta ronda
    model = get_model()
    model.model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Cargamos el split de validación completo 
    data_yaml = load_centralized_dataset()
    test_loss, map50 = test(model, data_yaml, device)

    # Devolvemos las métricas para que Flower las registre en los logs
    return MetricRecord({"map50": map50, "loss": test_loss})
