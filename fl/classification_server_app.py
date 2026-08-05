"""FedBN + FedYogi ServerApp for federated medical classification."""

import json
import logging
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.common import Array
from flwr.common.logger import log
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedYogi

from fl.classification_server_support import _clean_metrics
from fl.classification_task import (
    DATASET_CLASSES,
    DATASET_PROFILE,
    MultiDatasetRTDETRClassifier,
    confusions_from_metrics,
    load_round0_backbone,
    metrics_from_confusions,
    seed_everything,
    sha256_file,
)
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


class EarlyStopFederation(Exception):
    pass


class CheckpointingClassificationFedYogi(FedYogi):
    def __init__(self, *args, output_dir: Path, store: FedBNStateStore, run_meta: dict,
                 full_config: dict, round_offset: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dir = output_dir
        self.store = store
        self.run_meta = run_meta
        self.full_config = full_config
        self.round_offset = int(round_offset)
        self.latest_arrays = None
        self.train_by_round = {}
        self.history = []
        self.best_weighted_f1 = -math.inf
        self.best_accuracy = -math.inf
        self.best_round = 0
        self.current_round = 0
        self.early_stop_macro_f1_threshold = float(full_config.get("classification_early_stop_macro_f1_threshold", 0.0))
        self.early_stop_min_rounds = int(full_config.get("classification_early_stop_min_rounds", 1))
        self.early_stop_hit = False

    def _valid_or_fail(self, replies, *, is_train: bool):
        replies = list(replies)
        valid, invalid = self._check_and_log_replies(replies, is_train=is_train)
        if invalid or len(valid) != 5:
            raise RuntimeError(f"FedBN requires 5/5 valid clients: valid={len(valid)} invalid={len(invalid)}")
        return replies

    def aggregate_train(self, server_round, replies):
        replies = self._valid_or_fail(replies, is_train=True)
        # Flower 1.24 turns a zero-dimensional parameter (MoE gamma) into a
        # NumPy scalar during FedYogi. Preserve the exact FedYogi equations but
        # normalise every result back to ndarray before constructing Array.
        arrays, metrics = super(FedYogi, self).aggregate_train(server_round, replies)
        if arrays is None:
            raise RuntimeError("FedYogi returned no arrays")
        if self.current_arrays is None:
            raise RuntimeError("FedYogi current arrays were not initialised")
        delta_t, m_t, aggregated_ndarrays = self._compute_deltat_and_mt(arrays)
        if not self.v_t:
            self.v_t = {key: np.zeros_like(value) for key, value in aggregated_ndarrays.items()}
        self.v_t = {
            key: value - (1.0 - self.beta_2) * (delta_t[key] ** 2)
            * np.sign(value - delta_t[key] ** 2)
            for key, value in self.v_t.items()
        }
        new_arrays = {
            key: np.asarray(value + self.eta * m_t[key] / (np.sqrt(self.v_t[key]) + self.tau))
            for key, value in self.current_arrays.items()
        }
        arrays = ArrayRecord({key: Array(value) for key, value in new_arrays.items()})
        self.latest_arrays = arrays
        self.current_arrays = {name: array.numpy() for name, array in arrays.items()}
        client_rounds = {int(reply.content["metrics"]["actual_round"]) for reply in replies}
        if len(client_rounds) != 1:
            raise RuntimeError(f"Clients disagree on training round: {sorted(client_rounds)}")
        actual_round = client_rounds.pop()
        self.train_by_round[actual_round] = _clean_metrics(metrics or {})
        return arrays, metrics

    def aggregate_evaluate(self, server_round, replies):
        replies = self._valid_or_fail(replies, is_train=False)
        metrics = super().aggregate_evaluate(server_round, replies)
        records = [dict(reply.content["metrics"]) for reply in replies]
        exact = metrics_from_confusions(confusions_from_metrics(records))
        if metrics is not None:
            for key in [key for key in metrics if str(key).startswith("cm__")]: del metrics[key]
            for key, value in exact.items(): metrics[key] = float(value)
        client_rounds = {int(record["actual_round"]) for record in records}
        if len(client_rounds) != 1:
            raise RuntimeError(f"Clients disagree on evaluation round: {sorted(client_rounds)}")
        actual_round = client_rounds.pop()
        self.current_round = actual_round
        row = {"round": actual_round, **{f"train_{k}": v for k, v in self.train_by_round.get(actual_round, {}).items()}, **{f"val_{k}": v for k, v in _clean_metrics(exact).items()}}
        self.history.append(row)
        (self.output_dir / "metrics" / "classification_metrics_history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        if self.latest_arrays is None: raise RuntimeError("Missing latest shared arrays during evaluation")
        save_fedbn_bundle(self.output_dir / "latest_resume.pt", shared_arrays=self.latest_arrays,
                          store=self.store, strategy=self, round_num=actual_round,
                          metrics=_clean_metrics(exact), config=self.full_config)
        weighted_f1 = float(exact.get("weighted_f1", float("nan")))
        accuracy = float(exact.get("accuracy", float("nan")))
        if math.isfinite(weighted_f1) and weighted_f1 > self.best_weighted_f1:
            self.best_weighted_f1 = weighted_f1; self.best_round = actual_round
            save_fedbn_bundle(self.output_dir / "best_by_val_weighted_f1.pt", shared_arrays=self.latest_arrays,
                              store=self.store, strategy=self, round_num=actual_round,
                              metrics=_clean_metrics(exact), config=self.full_config)
        if math.isfinite(accuracy) and accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            save_fedbn_bundle(self.output_dir / "best_by_val_accuracy.pt", shared_arrays=self.latest_arrays,
                              store=self.store, strategy=self, round_num=actual_round,
                              metrics=_clean_metrics(exact), config=self.full_config)
        macro_f1 = float(exact.get("macro_f1", float("nan")))
        if (self.early_stop_macro_f1_threshold > 0.0
                and actual_round >= self.early_stop_min_rounds
                and math.isfinite(macro_f1)
                and macro_f1 >= self.early_stop_macro_f1_threshold):
            self.early_stop_hit = True
            raise EarlyStopFederation(
                f"validation macro-F1 {macro_f1:.6f} reached threshold "
                f"{self.early_stop_macro_f1_threshold:.6f} at round {actual_round}"
            )
        return metrics


@app.main()
def main(grid: Grid, context: Context) -> None:
    config = dict(context.run_config)
    if not as_bool(config.get("use_fedbn", False)):
        raise RuntimeError("FedBN ServerApp refuses use_fedbn=false")
    if str(config.get("server_optimizer", "")).lower() != "fedyogi":
        raise RuntimeError("FedBN experiment requires server_optimizer=fedyogi")
    master_seed = int(config.get("classification_master_seed", 42)); seed_everything(master_seed)
    rounds = int(config["num-server-rounds"]); num_clients = int(config.get("classification_num_clients", 5))
    if num_clients != 5: raise RuntimeError(f"Expected five clients, got {num_clients}")
    output_dir = Path(str(config["classification_output_dir"])); output_dir.mkdir(parents=True, exist_ok=False)
    for subdir in ("metrics", "configs", "shared_checkpoints", "logs"): (output_dir / subdir).mkdir()
    state_dir = Path(str(config["fedbn_state_dir"])); store = FedBNStateStore(state_dir, expected_clients=5)
    if any(state_dir.iterdir()): raise RuntimeError(f"FedBN state directory is not empty: {state_dir}")
    moe_enabled = as_bool(config.get("classification_moe_enabled", False))
    if not moe_enabled:
        raise RuntimeError("Classification MoE experiment requires classification_moe_enabled=true")
    model = MultiDatasetRTDETRClassifier(
        dropout=0.1,
        moe_enabled=True,
        moe_num_experts=int(config.get("classification_moe_num_experts", 4)),
        moe_top_k=int(config.get("classification_moe_top_k", 2)),
        moe_bottleneck=int(config.get("classification_moe_bottleneck", 256)),
        moe_gamma_init=float(config.get("classification_moe_gamma_init", 1e-3)),
    )
    round0_path = Path(str(config["classification_round0_path"]))
    load_report = load_round0_backbone(model, round0_path)
    if int(load_report["loaded"]) != 480: raise RuntimeError(f"Expected 480 SSL tensors, got {load_report}")
    model_layout = layout(model)
    expected_layouts = {
        "medical5": (105, 440),
        "fetal_planes": (97, 440),
        "cbis_ddsm": (97, 440),
        "nct_crc_he100k": (97, 440),
    }
    expected_layout = expected_layouts[DATASET_PROFILE]
    if (model_layout["shared_count"], model_layout["local_count"]) != expected_layout:
        raise RuntimeError(f"Classification FedBN layout changed: {model_layout}")
    initial_local_path = output_dir / "configs" / "round0_local_state.pt"; save_initial_local_state(model, initial_local_path)
    config.update({"fedbn_initial_local_state_path": str(initial_local_path), "round_offset": 0})
    run_meta = {
        "task": "classification", "bn_policy": "local_fedbn", "server_optimizer": "fedyogi",
        "model": "RTDETR_L_classification_sparse_MoE_profile_heads",
        "dataset_profile": DATASET_PROFILE, "dataset_heads": DATASET_CLASSES,
        "master_seed": master_seed, "rounds": rounds, "num_clients": 5,
        "round0_path": str(round0_path), "round0_sha256": sha256_file(round0_path),
        "round0_load_report": load_report, "fedbn_layout": model_layout,
        "data_root": str(config["classification_data_root"]),
        "manifest_sha256": str(config["classification_manifest_sha256"]),
        "stratified_manifest": str(config["classification_stratified_manifest"]),
        "stratified_manifest_sha256": str(config["classification_stratified_manifest_sha256"]),
        "local_epochs": int(config["classification_local_epochs"]),
        "backbone_lr": float(config["classification_backbone_lr"]), "head_lr": float(config["classification_head_lr"]),
        "moe_lr": float(config["classification_moe_lr"]),
        "server_eta": float(config["server_eta"]), "beta1": float(config["fedyogi_beta1"]),
        "beta2": float(config["fedyogi_beta2"]), "tau": float(config["fedyogi_tau"]),
        "selection_metric": "validation_weighted_f1", "moe_enabled": True,
        "moe_num_experts": int(config["classification_moe_num_experts"]),
        "moe_top_k": int(config["classification_moe_top_k"]),
        "moe_bottleneck": int(config["classification_moe_bottleneck"]),
        "moe_gamma_init": float(config["classification_moe_gamma_init"]),
        "moe_balance_loss_weight": float(config["classification_moe_balance_loss_weight"]),
        "code_version": os.environ.get("CODE_VERSION", "unknown"),
        "started_at": os.environ.get("EXPERIMENT_STARTED_AT", datetime.now(timezone.utc).isoformat()),
        "python_version": platform.python_version(), "torch_version": torch.__version__,
    }
    (output_dir / "configs" / "run_meta.json").write_text(json.dumps(run_meta, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "configs" / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    log(logging.INFO, "USE_FEDBN=True; BN policy=client local; Server aggregation=FedYogi; "
        f"shared_tensors={model_layout['shared_count']} local_tensors={model_layout['local_count']}")
    strategy = CheckpointingClassificationFedYogi(
        fraction_train=1.0, fraction_evaluate=1.0, min_train_nodes=5,
        min_evaluate_nodes=5, min_available_nodes=5, eta=float(config["server_eta"]),
        beta_1=float(config["fedyogi_beta1"]), beta_2=float(config["fedyogi_beta2"]),
        tau=float(config["fedyogi_tau"]), output_dir=output_dir, store=store,
        run_meta=run_meta, full_config=config,
    )
    stopped = False
    try:
        result = strategy.start(grid=grid, initial_arrays=shared_array_record(model),
                                train_config=ConfigRecord(config), evaluate_config=ConfigRecord(config), num_rounds=rounds)
    except EarlyStopFederation as exc:
        stopped = True
        log(logging.INFO, str(exc))
    if strategy.latest_arrays is None: raise RuntimeError("Training completed without shared arrays")
    completed_rounds = int(strategy.current_round)
    save_fedbn_bundle(output_dir / "final.pt", shared_arrays=strategy.latest_arrays, store=store,
                      strategy=strategy, round_num=completed_rounds,
                      metrics={"best_weighted_f1": strategy.best_weighted_f1, "best_accuracy": strategy.best_accuracy}, config=config)
    (output_dir / "server_result.json").write_text(json.dumps({**run_meta, "early_stopped": stopped, "early_stop_macro_f1_threshold": strategy.early_stop_macro_f1_threshold, "completed_rounds": completed_rounds, "best_round": strategy.best_round, "best_weighted_f1": strategy.best_weighted_f1, "best_accuracy": strategy.best_accuracy, "best_checkpoint": str(output_dir / "best_by_val_weighted_f1.pt"), "final_checkpoint": str(output_dir / "final.pt")}, indent=2, sort_keys=True), encoding="utf-8")
