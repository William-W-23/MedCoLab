"""FedBN + FedYogi ServerApp for RT-DETR-L + ASEM detection."""

import json
import logging
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from flwr.app import Array, ArrayRecord, ConfigRecord, Context
from flwr.common.logger import log
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedYogi

from fl.detection_server_core import Net, FL_MODEL_VARIANT, pretrained_model_path, ssl_backbone_path, _metric_record_to_plain
from fl.finetune_from_ssl_task import load_ssl_backbone_weights
from fl.detection_task import DATASET_CONFIGS, CURRENT_DATASET, load_rtdetr_weights
from fl.fedbn_runtime import (
    FedBNStateStore,
    as_bool,
    layout,
    restore_strategy_state,
    save_fedbn_bundle,
    save_initial_local_state,
    shared_array_record,
)

app = ServerApp()


class CheckpointingDetectionFedYogi(FedYogi):
    def __init__(self, *args, output_dir: Path, store: FedBNStateStore, full_config: dict,
                 round_offset: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dir = output_dir; self.store = store; self.full_config = full_config
        self.round_offset = int(round_offset); self.latest_arrays = None
        self.best_metric_value = float("-inf"); self.best_metric_round = 0; self.history = []

    def _valid_or_fail(self, replies, *, is_train):
        replies = list(replies); valid, invalid = self._check_and_log_replies(replies, is_train=is_train)
        if invalid or len(valid) != 5: raise RuntimeError(f"FedBN requires 5/5 valid clients: valid={len(valid)} invalid={len(invalid)}")
        return replies

    def aggregate_train(self, server_round, replies):
        replies = self._valid_or_fail(replies, is_train=True)
        # Flower 1.24 turns a zero-dimensional NumPy array into a NumPy scalar
        # while updating ASEM's scalar gamma. Array rejects that scalar even
        # though the optimizer math is valid. Keep the FedYogi equations and
        # normalize every result back to ndarray before constructing Array.
        averaged, metrics = FedAvg.aggregate_train(self, server_round, replies)
        if averaged is None or self.current_arrays is None:
            raise RuntimeError("FedYogi returned no arrays/current state")
        delta_t, m_t, averaged_ndarrays = self._compute_deltat_and_mt(averaged)
        if not self.v_t:
            self.v_t = {key: np.zeros_like(value) for key, value in averaged_ndarrays.items()}
        self.v_t = {
            key: value - (1.0 - self.beta_2) * (delta_t[key] ** 2)
            * np.sign(value - delta_t[key] ** 2)
            for key, value in self.v_t.items()
        }
        updated = {
            key: np.asarray(value + self.eta * m_t[key] / (np.sqrt(self.v_t[key]) + self.tau))
            for key, value in self.current_arrays.items()
        }
        arrays = ArrayRecord({key: Array(value) for key, value in updated.items()})
        self.latest_arrays = arrays; self.current_arrays = {name: array.numpy() for name, array in arrays.items()}
        client_rounds = {int(reply.content["metrics"]["actual_round"]) for reply in replies}
        if len(client_rounds) != 1:
            raise RuntimeError(f"Clients disagree on training round: {sorted(client_rounds)}")
        self.latest_actual_round = client_rounds.pop()
        return arrays, metrics

    def aggregate_evaluate(self, server_round, replies):
        replies = self._valid_or_fail(replies, is_train=False)
        metrics = super().aggregate_evaluate(server_round, replies)
        if metrics is None or self.latest_arrays is None: raise RuntimeError("Missing evaluation metrics/shared arrays")
        records = [dict(reply.content["metrics"]) for reply in replies]
        client_rounds = {int(record["actual_round"]) for record in records}
        if len(client_rounds) != 1:
            raise RuntimeError(f"Clients disagree on evaluation round: {sorted(client_rounds)}")
        actual_round = client_rounds.pop(); plain = _metric_record_to_plain(metrics)
        self.history.append({"round": actual_round, **plain})
        (self.output_dir / "metrics" / "metrics_history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        save_fedbn_bundle(self.output_dir / "latest_resume.pt", shared_arrays=self.latest_arrays,
                          store=self.store, strategy=self, round_num=actual_round, metrics=plain, config=self.full_config)
        value = float(plain.get("eval_map50", float("-inf")))
        if value > self.best_metric_value:
            self.best_metric_value = value; self.best_metric_round = actual_round
            save_fedbn_bundle(self.output_dir / "best_by_val_map50.pt", shared_arrays=self.latest_arrays,
                              store=self.store, strategy=self, round_num=actual_round, metrics=plain, config=self.full_config)
        return metrics


@app.main()
def main(grid: Grid, context: Context) -> None:
    config = dict(context.run_config)
    if not as_bool(config.get("use_fedbn", False)): raise RuntimeError("FedBN ServerApp refuses use_fedbn=false")
    if str(config.get("server_optimizer", "")).lower() != "fedyogi": raise RuntimeError("Expected server_optimizer=fedyogi")
    if int(config.get("num_clients", 5)) != 5: raise RuntimeError("Detection experiment requires five clients")
    seed = int(config.get("master_seed", os.environ.get("MASTER_SEED", 42)))
    os.environ["PYTHONHASHSEED"] = str(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    resume_path_text = str(config.get("resume_bundle_path", "")).strip()
    resume_payload = None
    round_offset = 0
    if resume_path_text:
        resume_path = Path(resume_path_text)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Missing FedBN resume bundle: {resume_path}")
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_payload.get("schema") != "fedbn_fedyogi_bundle_v1":
            raise RuntimeError(f"Unsupported resume schema: {resume_payload.get('schema')}")
        round_offset = int(resume_payload["round"])
        expected_total = int(config.get("resume_total_rounds", 0))
        if expected_total <= round_offset:
            raise RuntimeError(
                f"resume_total_rounds={expected_total} must exceed bundle round={round_offset}"
            )
        if round_offset + int(config["num-server-rounds"]) != expected_total:
            raise RuntimeError(
                "Resume round mismatch: "
                f"offset={round_offset} remaining={config['num-server-rounds']} total={expected_total}"
            )
        prior = dict(resume_payload.get("config", {}))
        for key in ("lr", "master_seed", "server_eta", "fedyogi_beta1", "fedyogi_beta2", "fedyogi_tau"):
            if str(prior.get(key)) != str(config.get(key)):
                raise RuntimeError(f"Resume config mismatch for {key}: {prior.get(key)} != {config.get(key)}")

    output_dir = Path(str(config["output_dir"])); output_dir.mkdir(parents=True, exist_ok=resume_payload is not None)
    for subdir in ("metrics", "configs", "shared_checkpoints", "logs"):
        (output_dir / subdir).mkdir(exist_ok=resume_payload is not None)
    state_dir = Path(str(config["fedbn_state_dir"])); store = FedBNStateStore(state_dir, expected_clients=5)
    if resume_payload is None:
        if any(state_dir.iterdir()): raise RuntimeError(f"FedBN state directory is not empty: {state_dir}")
    else:
        client_states = resume_payload.get("client_local_state", {})
        if {int(key) for key in client_states} != set(range(5)):
            raise RuntimeError("Resume bundle does not contain client0..4 local states")
        for client_id in range(5):
            state = client_states.get(client_id, client_states.get(str(client_id)))
            store.save(client_id, round_offset, state)
    nc = DATASET_CONFIGS[CURRENT_DATASET]["nc"]; model = Net(nc=nc)
    load_rtdetr_weights(model, pretrained_model_path)
    ssl_loaded = load_ssl_backbone_weights(model, ssl_backbone_path)
    model_layout = layout(model)
    if (model_layout["shared_count"], model_layout["local_count"]) != (348, 671):
        raise RuntimeError(f"Detection FedBN layout changed: {model_layout}")
    initial_name = "round0_local_state.pt" if resume_payload is None else f"resume_round{round_offset}_initial_local_state.pt"
    initial_local_path = output_dir / "configs" / initial_name; save_initial_local_state(model, initial_local_path)
    config.update({"fedbn_initial_local_state_path": str(initial_local_path), "round_offset": round_offset,
                   "moe_domain_supervision": False})
    logical_rounds = round_offset + int(config["num-server-rounds"])
    run_meta = {
        "task": "detection", "model": FL_MODEL_VARIANT, "bn_policy": "local_fedbn",
        "server_optimizer": "fedyogi", "num_clients": 5, "rounds": logical_rounds,
        "local_epochs": int(config["local-epochs"]), "client_lr": float(config["lr"]),
        "server_eta": float(config["server_eta"]), "beta1": float(config["fedyogi_beta1"]),
        "beta2": float(config["fedyogi_beta2"]), "tau": float(config["fedyogi_tau"]),
        "seed": seed, "dataset": CURRENT_DATASET, "pretrained_model_path": pretrained_model_path,
        "ssl_backbone_path": ssl_backbone_path, "ssl_loaded_tensors": ssl_loaded,
        "fedbn_layout": model_layout, "moe_enabled": True,
        "moe_domain_supervision": "disabled_invalid_client_id_mapping_task_driven_router",
        "selection_metric": "validation_mAP50", "code_version": os.environ.get("CODE_VERSION", "unknown"),
        "started_at": os.environ.get("EXPERIMENT_STARTED_AT", datetime.now(timezone.utc).isoformat()),
        "python_version": platform.python_version(), "torch_version": torch.__version__,
        "resumed_from": resume_path_text or None, "round_offset": round_offset,
    }
    meta_name = "run_meta.json" if resume_payload is None else f"run_meta_resume_round{round_offset}.json"
    config_name = "run_config.json" if resume_payload is None else f"run_config_resume_round{round_offset}.json"
    (output_dir / "configs" / meta_name).write_text(json.dumps(run_meta, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "configs" / config_name).write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    log(logging.INFO, "USE_FEDBN=True; BN policy=client local; Server aggregation=FedYogi; "
        f"shared_tensors={model_layout['shared_count']} local_tensors={model_layout['local_count']}")
    strategy = CheckpointingDetectionFedYogi(
        fraction_train=1.0, fraction_evaluate=1.0, min_train_nodes=5, min_evaluate_nodes=5,
        min_available_nodes=5, eta=float(config["server_eta"]), beta_1=float(config["fedyogi_beta1"]),
        beta_2=float(config["fedyogi_beta2"]), tau=float(config["fedyogi_tau"]),
        output_dir=output_dir, store=store, full_config=config,
        round_offset=round_offset,
    )
    rounds = int(config["num-server-rounds"])
    initial_arrays = shared_array_record(model)
    if resume_payload is not None:
        initial_arrays = ArrayRecord(resume_payload["shared_state"])
        restore_strategy_state(strategy, resume_payload["server_optimizer_state"])
        strategy.latest_arrays = initial_arrays
        history_path = output_dir / "metrics" / "metrics_history.json"
        if history_path.is_file():
            strategy.history = json.loads(history_path.read_text(encoding="utf-8"))
            if strategy.history:
                prior_best = max(strategy.history, key=lambda row: float(row.get("eval_map50", float("-inf"))))
                strategy.best_metric_value = float(prior_best["eval_map50"])
                strategy.best_metric_round = int(prior_best["round"])
        log(logging.INFO, f"Resuming FedBN+FedYogi from round {round_offset} for {rounds} remaining rounds")
    result = strategy.start(grid=grid, initial_arrays=initial_arrays, train_config=ConfigRecord(config),
                            evaluate_config=ConfigRecord(config), num_rounds=rounds)
    save_fedbn_bundle(output_dir / "final.pt", shared_arrays=result.arrays, store=store,
                      strategy=strategy, round_num=logical_rounds,
                      metrics={"best_map50": strategy.best_metric_value}, config=config)
    (output_dir / "server_result.json").write_text(json.dumps({**run_meta, "completed_rounds": logical_rounds,
        "best_round": strategy.best_metric_round, "best_map50": strategy.best_metric_value,
        "best_checkpoint": str(output_dir / "best_by_val_map50.pt"), "final_checkpoint": str(output_dir / "final.pt")},
        indent=2, sort_keys=True), encoding="utf-8")
