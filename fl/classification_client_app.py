"""FedBN ClientApp for the five-head medical classification experiment."""

import logging
import os
from pathlib import Path

import torch
from flwr.app import Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.logger import log

from fl.classification_client_support import build_stratified_loader, _config
from fl.classification_task import (
    MultiDatasetRTDETRClassifier,
    evaluate,
    seed_everything,
    train_one_round,
)
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


def _build_fedbn_model(config: dict, msg: Message, client_id: int, actual_round: int, device):
    if not as_bool(config.get("use_fedbn", False)):
        raise RuntimeError("FedBN ClientApp refuses use_fedbn=false")
    model = MultiDatasetRTDETRClassifier(
        dropout=0.1,
        moe_enabled=as_bool(config.get("classification_moe_enabled", False)),
        moe_num_experts=int(config.get("classification_moe_num_experts", 4)),
        moe_top_k=int(config.get("classification_moe_top_k", 2)),
        moe_bottleneck=int(config.get("classification_moe_bottleneck", 256)),
        moe_gamma_init=float(config.get("classification_moe_gamma_init", 1e-3)),
    ).to(device)
    initial_path = Path(str(config["fedbn_initial_local_state_path"]))
    if not initial_path.is_file():
        raise FileNotFoundError(f"Missing Round-0 local BN state: {initial_path}")
    load_initial_local_state(model, initial_path)
    load_shared_array_record(model, msg.content["arrays"])
    store = FedBNStateStore(str(config["fedbn_state_dir"]), expected_clients=5)
    state, saved_round = store.load(client_id)
    if state is not None:
        expected_previous = actual_round - 1
        if saved_round not in {expected_previous, actual_round}:
            raise RuntimeError(
                f"client{client_id} stale BN state: saved_round={saved_round}, "
                f"expected={expected_previous} or {actual_round}"
            )
        load_local_state(model, state)
        source = f"client-local round {saved_round}"
    else:
        if actual_round != 1:
            raise RuntimeError(f"client{client_id} has no BN state before round {actual_round}")
        source = "SSL Round-0 initial BN"
    model_layout = layout(model)
    log(logging.INFO, f"[Classification client{client_id}] FedBN restore={source} "
        f"shared_tensors={model_layout['shared_count']} local_tensors={model_layout['local_count']}")
    return model, store


@app.train()
def train(msg: Message, context: Context):
    config = _config(context, msg)
    client_id = int(context.node_config["partition-id"])
    internal_round = int(config["server-round"])
    actual_round = int(config.get("round_offset", 0)) + internal_round
    master_seed = int(config.get("classification_master_seed", os.environ.get("MASTER_SEED", 42)))
    seed_everything(master_seed + client_id * 10_000 + actual_round)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, store = _build_fedbn_model(config, msg, client_id, actual_round, device)
    loader, stats = build_stratified_loader(config, client_id, "train", actual_round)
    log(logging.INFO, f"[Classification client{client_id}] actual_round={actual_round} train={stats}")
    metrics = train_one_round(
        model,
        loader,
        device=device,
        epochs=int(config["classification_local_epochs"]),
        lr=float(config["classification_lr"]),
        max_batches=int(config.get("classification_train_max_batches", 0)),
        backbone_lr=float(config.get("classification_backbone_lr", config["classification_lr"])),
        head_lr=float(config.get("classification_head_lr", config["classification_lr"])),
        weight_decay=float(config.get("classification_weight_decay", 1e-4)),
        label_smoothing=float(config.get("classification_label_smoothing", 0.0)),
        class_weight_power=float(config.get("classification_class_weight_power", 1.0)),
        moe_lr=float(config.get("classification_moe_lr", config["classification_lr"])),
        moe_balance_loss_weight=float(config.get("classification_moe_balance_loss_weight", 0.01)),
    )
    local_state = capture_local_state(model)
    store.save(client_id, actual_round, local_state)
    metrics.update({
        "num-examples": len(loader.dataset),
        "fedbn_local_tensors": len(local_state),
        "actual_round": actual_round,
    })
    return Message(
        content=RecordDict({"arrays": shared_array_record(model), "metrics": MetricRecord(metrics)}),
        reply_to=msg,
    )


@app.evaluate()
def evaluate_client(msg: Message, context: Context):
    config = _config(context, msg)
    client_id = int(context.node_config["partition-id"])
    internal_round = int(config["server-round"])
    actual_round = int(config.get("round_offset", 0)) + internal_round
    master_seed = int(config.get("classification_master_seed", os.environ.get("MASTER_SEED", 42)))
    seed_everything(master_seed + client_id * 10_000 + 100_000 + actual_round)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, store = _build_fedbn_model(config, msg, client_id, actual_round, device)
    _, saved_round = store.load(client_id)
    if saved_round != actual_round:
        raise RuntimeError(f"client{client_id} evaluation BN round={saved_round}, expected={actual_round}")
    loader, stats = build_stratified_loader(config, client_id, "val", actual_round)
    log(logging.INFO, f"[Classification client{client_id}] actual_round={actual_round} val={stats}")
    metrics = evaluate(model, loader, device=device, max_batches=int(config.get("classification_eval_max_batches", 0)))
    metrics.update({"num-examples": len(loader.dataset), "actual_round": actual_round})
    return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)
