"""pytorchexample: lógica del cliente federado para detección de matrículas."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import get_model, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

# Creamos la aplicación cliente de Flower
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Entrena el modelo localmente con los datos de este cliente."""

    # Cargamos YOLOv8n y le cargamos los pesos globales que nos manda el servidor
    model = get_model()
    model.model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Cada cliente tiene un id distinto para saber qué trozo del dataset le toca
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data_yaml, num_train, _ = load_data(partition_id, num_partitions)

    # Entrenamos el modelo con los datos locales
    model, train_loss = train_fn(
        model,
        data_yaml,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],  # tasa de aprendizaje que nos pasa el servidor
        device,
    )

    # Enviamos de vuelta los pesos actualizados y la pérdida al servidor
    model_record = ArrayRecord(model.model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": num_train,  # necesario para que FedAvg pondere correctamente
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evalúa el modelo global con los datos locales de este cliente."""

    # Cargamos YOLOv8n con los pesos globales recibidos del servidor
    model = get_model()
    model.model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Cargamos la partición de datos de este cliente
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data_yaml, _, num_val = load_data(partition_id, num_partitions)

    # Evaluamos el modelo y obtenemos la pérdida y el mAP50
    eval_loss, map50 = test_fn(model, data_yaml, device)

    # Devolvemos las métricas al servidor (sin pesos, solo resultados)
    metrics = {
        "eval_loss": eval_loss,
        "map50": map50,
        "num-examples": num_val,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
