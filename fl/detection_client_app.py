"""FedBN ClientApp for RT-DETR-L + ASEM detection."""

import logging
import os
from pathlib import Path

import torch
from flwr.app import Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.logger import log

from fl.detection_client_core import Net, seed_client, validate_mode_compatibility
from fl.detection_task import DATASET_CONFIGS, CURRENT_DATASET, load_data, test as test_fn, train as train_fn
from fl.fedbn_runtime import (
    FedBNStateStore,
    as_bool,
    capture_local_state,
    layout,
    load_initial_local_state,
    load_local_state,
    load_shared_array_record,
    shared_array_record,
)

app = ClientApp()


def _build_model(config, msg, client_id, actual_round, device):
    if not as_bool(config.get("use_fedbn", False)):
        raise RuntimeError("FedBN ClientApp refuses use_fedbn=false")
    nc = DATASET_CONFIGS[CURRENT_DATASET]["nc"]
    model = Net(nc=nc).to(device)
    initial_path = Path(str(config["fedbn_initial_local_state_path"]))
    if not initial_path.is_file():
        raise FileNotFoundError(f"Missing SSL Round-0 local state: {initial_path}")
    load_initial_local_state(model, initial_path)
    load_shared_array_record(model, msg.content["arrays"])
    store = FedBNStateStore(str(config["fedbn_state_dir"]), expected_clients=5)
    state, saved_round = store.load(client_id)
    if state is not None:
        if saved_round not in {actual_round - 1, actual_round}:
            raise RuntimeError(f"client{client_id} stale BN round={saved_round}, actual={actual_round}")
        load_local_state(model, state); source = f"client-local round {saved_round}"
    else:
        if actual_round != 1: raise RuntimeError(f"client{client_id} missing BN before round {actual_round}")
        source = "SSL Round-0 initial BN"
    model_layout = layout(model)
    log(logging.INFO, f"[Detection client{client_id}] FedBN restore={source} "
        f"shared_tensors={model_layout['shared_count']} local_tensors={model_layout['local_count']}")
    return model, store, nc


def _moe_diagnostics(model):
    router_grad = 0.0; expert_grad = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None: continue
        value = float(parameter.grad.detach().abs().sum().item())
        if "asem_p5.router" in name: router_grad += value
        if "asem_p5.experts" in name: expert_grad += value
    entropy = 0.0; usage = [0.0, 0.0, 0.0, 0.0]
    asem = getattr(model, "asem_p5", None)
    if asem is not None and asem.last_router_probs is not None:
        probs = asem.last_router_probs.float().clamp_min(1e-8)
        entropy = float((-(probs * probs.log()).sum(dim=1).mean()).item())
    if asem is not None and asem.last_routing_weights is not None:
        usage = [float(v) for v in asem.last_routing_weights.float().mean(dim=0).cpu().tolist()]
    return router_grad, expert_grad, entropy, usage


def _config(msg: Message, context: Context) -> dict:
    config = dict(context.run_config); config.update(dict(msg.content.get("config", {}))); return config


@app.train()
def train(msg: Message, context: Context):
    config = _config(msg, context)
    client_id = int(context.node_config["partition-id"])
    internal_round = int(config["server-round"]); actual_round = int(config.get("round_offset", 0)) + internal_round
    seed_client(client_id, actual_round)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, store, nc = _build_model(config, msg, client_id, actual_round, device)
    trainloader, _, _, _ = load_data(client_id, int(context.node_config["num-partitions"]))
    loss = train_fn(model, trainloader, int(config["local-epochs"]), float(config["lr"]), device,
                    nc=nc, moe_domain_supervision=as_bool(config.get("moe_domain_supervision", False)),
                    max_batches=int(config.get("train_max_batches", 0)))
    router_grad, expert_grad, entropy, usage = _moe_diagnostics(model)
    if router_grad <= 0 or expert_grad <= 0:
        raise RuntimeError(f"MoE gradient check failed: router={router_grad}, experts={expert_grad}")
    local_state = capture_local_state(model); store.save(client_id, actual_round, local_state)
    metrics = {
        "train_loss": float(loss), "num-examples": len(trainloader.dataset),
        "actual_round": actual_round, "fedbn_local_tensors": len(local_state),
        "router_grad_l1": router_grad, "expert_grad_l1": expert_grad,
        "router_entropy": entropy,
        **{f"expert_usage_{idx}": value for idx, value in enumerate(usage)},
    }
    return Message(content=RecordDict({"arrays": shared_array_record(model), "metrics": MetricRecord(metrics)}), reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    config = _config(msg, context)
    client_id = int(context.node_config["partition-id"])
    internal_round = int(config["server-round"]); actual_round = int(config.get("round_offset", 0)) + internal_round
    seed_client(client_id, actual_round)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, store, nc = _build_model(config, msg, client_id, actual_round, device)
    _, saved_round = store.load(client_id)
    if saved_round != actual_round: raise RuntimeError(f"client{client_id} evaluation BN round={saved_round}, expected={actual_round}")
    _, valloader, _, _ = load_data(client_id, int(context.node_config["num-partitions"]))
    loss, map50, precision, recall, f1, _ = test_fn(
        model, valloader, device, nc=nc, client_id=client_id, split_name="val",
        moe_domain_supervision=as_bool(config.get("moe_domain_supervision", False)),
        max_batches=int(config.get("eval_max_batches", 0)),
    )
    def safe(value):
        value = float(value); return value if value == value and value >= 0 else 0.0
    metrics = {"eval_loss": safe(loss), "eval_map50": safe(map50), "eval_precision": safe(precision),
               "eval_recall": safe(recall), "eval_f1": safe(f1), "num-examples": len(valloader.dataset),
               "actual_round": actual_round}
    return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)
