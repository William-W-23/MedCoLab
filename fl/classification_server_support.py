"""Flower ServerApp for federated multi-dataset medical classification."""

import json
import logging
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.common.logger import log
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fl.classification_task import (
    DATASET_CLASSES,
    MultiDatasetRTDETRClassifier,
    confusions_from_metrics,
    load_round0_backbone,
    metrics_from_confusions,
    seed_everything,
    sha256_file,
)

app = ServerApp()


def _clean_metrics(metrics: dict) -> dict:
    return {str(k): float(v) for k, v in metrics.items() if not str(k).startswith("cm__") and isinstance(v, (int, float))}


class ClassificationFedAvg(FedAvg):
    def __init__(self, *args, model, output_dir: Path, run_meta: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.output_dir = output_dir
        self.run_meta = run_meta
        self.history = []
        self.train_by_round = {}
        self.best_pcam_macro_f1 = -math.inf
        self.best_pcam_accuracy = -math.inf
        self.best_overall_accuracy = -math.inf
        self.best_weighted_f1 = -math.inf

    def _save_model(self, filename: str, round_num: int, metrics: dict) -> None:
        path = self.output_dir / filename
        torch.save(self.model.state_dict(), path)
        meta = {
            **self.run_meta,
            "checkpoint": str(path),
            "round": int(round_num),
            "metrics": _clean_metrics(metrics),
            "checkpoint_sha256": sha256_file(path),
        }
        (self.output_dir / filename.replace(".pt", "_meta.json")).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    def aggregate_train(self, server_round, replies):
        reply_list = list(replies)
        arrays, metrics = super().aggregate_train(server_round, reply_list)
        if arrays is not None:
            self.model.load_state_dict(arrays.to_torch_state_dict(), strict=True)
        self.train_by_round[int(server_round)] = _clean_metrics(metrics or {})
        return arrays, metrics

    def aggregate_evaluate(self, server_round, replies):
        reply_list = list(replies)
        metrics = super().aggregate_evaluate(server_round, reply_list)
        records = [dict(reply.content["metrics"]) for reply in reply_list if reply.content.get("metrics") is not None]
        exact = metrics_from_confusions(confusions_from_metrics(records))
        if metrics is not None:
            for key in [key for key in metrics if str(key).startswith("cm__")]:
                del metrics[key]
            for key, value in exact.items():
                metrics[key] = float(value)
        row = {
            "round": int(server_round),
            **{f"train_{k}": v for k, v in self.train_by_round.get(int(server_round), {}).items()},
            **{f"val_{k}": v for k, v in _clean_metrics(exact).items()},
        }
        self.history.append(row)
        (self.output_dir / "classification_metrics_history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        pcam_f1 = float(exact.get("pcam_macro_f1", float("nan")))
        pcam_acc = float(exact.get("pcam_accuracy", float("nan")))
        if math.isfinite(pcam_f1) and pcam_f1 > self.best_pcam_macro_f1:
            self.best_pcam_macro_f1 = pcam_f1
            self._save_model("best_by_pcam_val_macro_f1.pt", server_round, exact)
            log(logging.INFO, f"Saved best PCam Macro-F1: round={server_round}, f1={pcam_f1:.6f}")
        if math.isfinite(pcam_acc) and pcam_acc > self.best_pcam_accuracy:
            self.best_pcam_accuracy = pcam_acc
            self._save_model("best_by_pcam_val_accuracy.pt", server_round, exact)
            log(logging.INFO, f"Saved best PCam accuracy: round={server_round}, acc={pcam_acc:.6f}")
        overall_acc=float(exact.get("accuracy",float("nan"))); weighted_f1=float(exact.get("weighted_f1",float("nan")))
        if math.isfinite(overall_acc) and overall_acc>self.best_overall_accuracy:
            self.best_overall_accuracy=overall_acc; self._save_model("best_by_val_accuracy.pt",server_round,exact)
        if math.isfinite(weighted_f1) and weighted_f1>self.best_weighted_f1:
            self.best_weighted_f1=weighted_f1; self._save_model("best_by_val_weighted_f1.pt",server_round,exact)
        return metrics


@app.main()
def main(grid: Grid, context: Context) -> None:
    config = dict(context.run_config)
    master_seed = int(config.get("classification_master_seed", os.environ.get("MASTER_SEED", 42)))
    seed_everything(master_seed)
    config["classification_master_seed"] = master_seed
    rounds = int(config["num-server-rounds"])
    num_clients = int(config.get("classification_num_clients", 5))
    output_dir = Path(str(config["classification_output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    round0_path = Path(str(config["classification_round0_path"]))
    if not round0_path.is_file():
        raise FileNotFoundError(f"Missing Round 0: {round0_path}")
    model = MultiDatasetRTDETRClassifier(dropout=0.1)
    load_report = load_round0_backbone(model, round0_path)
    if int(load_report["loaded"]) != 480:
        raise RuntimeError(f"Expected 480 SSL backbone tensors, got {load_report}")
    run_meta = {
        "task": "medical5_federated_multitask_classification_stratified_v5",
        "model": "RTDETR_L_shared_backbone_five_dataset_heads",
        "dataset_heads": DATASET_CLASSES,
        "eyepacs_supervised_classes": 4,
        "selection_scope": "primary_overall_weighted_f1_and_accuracy_on_feature_balanced_val_no_pcam_selection",
        "master_seed": master_seed,
        "manifest_sha256": str(config["classification_manifest_sha256"]),
        "stratified_manifest": str(config["classification_stratified_manifest"]),
        "stratified_manifest_sha256": str(config["classification_stratified_manifest_sha256"]),
        "data_root": str(config["classification_data_root"]),
        "round0_path": str(round0_path),
        "round0_sha256": sha256_file(round0_path),
        "round0_load_report": load_report,
        "rounds": rounds,
        "local_epochs": int(config["classification_local_epochs"]),
        "lr": float(config["classification_lr"]),
        "backbone_lr": float(config.get("classification_backbone_lr", config["classification_lr"])),
        "head_lr": float(config.get("classification_head_lr", config["classification_lr"])),
        "weight_decay": float(config.get("classification_weight_decay", 1e-4)),
        "label_smoothing": float(config.get("classification_label_smoothing", 0.0)),
        "class_weight_power": float(config.get("classification_class_weight_power", 1.0)),
        "pcam_group_balanced": bool(config.get("classification_pcam_group_balanced", False)),
        "pcam_augmentation": "full_field_resize_hflip_vflip_mild_colorjitter_no_random_resized_crop",
        "batch_size": int(config["classification_batch_size"]),
        "eval_batch_size": int(config["classification_eval_batch_size"]),
        "image_size": int(config["classification_image_size"]),
        "code_version": os.environ.get("CODE_VERSION", "unknown"),
        "started_at": os.environ.get("EXPERIMENT_STARTED_AT", datetime.now(timezone.utc).isoformat()),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    (output_dir / "classification_run_meta.json").write_text(json.dumps(run_meta, indent=2, sort_keys=True), encoding="utf-8")
    strategy = ClassificationFedAvg(
        fraction_train=1.0, fraction_evaluate=1.0,
        min_train_nodes=num_clients, min_evaluate_nodes=num_clients, min_available_nodes=num_clients,
        model=model, output_dir=output_dir, run_meta=run_meta,
    )
    shared_config = ConfigRecord(config)
    result = strategy.start(
        grid=grid, initial_arrays=ArrayRecord(model.state_dict()),
        train_config=shared_config, evaluate_config=shared_config, num_rounds=rounds,
    )
    model.load_state_dict(result.arrays.to_torch_state_dict(), strict=True)
    final_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    server_result = {
        **run_meta,
        "final_model": str(final_path),
        "final_model_sha256": sha256_file(final_path),
        "best_pcam_macro_f1": strategy.best_pcam_macro_f1,
        "best_pcam_accuracy": strategy.best_pcam_accuracy,
        "best_overall_accuracy": strategy.best_overall_accuracy,
        "best_weighted_f1": strategy.best_weighted_f1,
        "completed_rounds": rounds,
    }
    (output_dir / "server_result.json").write_text(json.dumps(server_result, indent=2, sort_keys=True), encoding="utf-8")
