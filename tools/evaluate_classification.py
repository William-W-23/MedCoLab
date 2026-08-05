#!/usr/bin/env python3
"""Evaluate five validation-selected personalized classifiers once on fixed test data."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from fl.classification_client_support import build_stratified_loader
from fl.classification_task import (
    DATASET_CLASSES,
    DATASET_NAMES,
    MultiDatasetRTDETRClassifier,
    empty_confusions,
    metrics_from_confusions,
    seed_everything,
    sha256_file,
)


def binary_auroc_auprc(binary: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """Compute tie-aware ROC-AUC and average precision from descending thresholds."""
    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_target = binary[order].astype(np.float64)
    threshold_indices = np.r_[np.where(np.diff(sorted_score))[0], sorted_target.size - 1]
    true_positives = np.cumsum(sorted_target)[threshold_indices]
    false_positives = (1 + threshold_indices) - true_positives
    positives = float(sorted_target.sum())
    negatives = float(sorted_target.size - positives)
    tpr = np.r_[0.0, true_positives / positives]
    fpr = np.r_[0.0, false_positives / negatives]
    auroc = float(np.trapz(tpr, fpr))
    precision = true_positives / np.maximum(true_positives + false_positives, 1.0)
    recall = true_positives / positives
    auprc = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return auroc, auprc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, nargs=5, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--moe-enabled", action="store_true")
    parser.add_argument("--moe-num-experts", type=int, default=4)
    parser.add_argument("--moe-top-k", type=int, default=2)
    parser.add_argument("--moe-bottleneck", type=int, default=256)
    parser.add_argument("--moe-gamma-init", type=float, default=1e-3)
    parser.add_argument("--selection-description", required=True)
    return parser.parse_args()


def probability_metrics(targets: dict[str, list[np.ndarray]], scores: dict[str, list[np.ndarray]]) -> dict:
    class_rows = []
    result = {}
    for dataset in DATASET_NAMES:
        if not targets[dataset]:
            continue
        y = np.concatenate(targets[dataset], axis=0)
        prob = np.concatenate(scores[dataset], axis=0)
        dataset_rows = []
        for class_id, label in enumerate(DATASET_CLASSES[dataset]):
            binary = (y == class_id).astype(np.int64)
            support = int(binary.sum())
            if support == 0 or support == len(binary):
                continue
            auroc, auprc = binary_auroc_auprc(binary, prob[:, class_id])
            row = {
                "dataset": dataset,
                "class": label,
                "support": support,
                "auroc": auroc,
                "auprc": auprc,
            }
            dataset_rows.append(row)
            class_rows.append(row)
        if dataset_rows:
            weight = sum(row["support"] for row in dataset_rows)
            result[f"{dataset}_weighted_auroc"] = sum(row["auroc"] * row["support"] for row in dataset_rows) / weight
            result[f"{dataset}_weighted_auprc"] = sum(row["auprc"] * row["support"] for row in dataset_rows) / weight
    total_support = sum(row["support"] for row in class_rows)
    result["macro_auroc"] = sum(row["auroc"] for row in class_rows) / max(len(class_rows), 1)
    result["macro_auprc"] = sum(row["auprc"] for row in class_rows) / max(len(class_rows), 1)
    result["weighted_auroc"] = sum(row["auroc"] * row["support"] for row in class_rows) / total_support
    result["weighted_auprc"] = sum(row["auprc"] * row["support"] for row in class_rows) / total_support
    result["probability_metric_definition"] = (
        "one-vs-rest AUROC/AUPRC for every valid class, weighted by class test support "
        "across the active dataset profile"
    )
    return result


@torch.no_grad()
def evaluate_once(model: torch.nn.Module, loader, device: torch.device):
    model.eval()
    confusions = empty_confusions()
    targets = {dataset: [] for dataset in DATASET_NAMES}
    scores = {dataset: [] for dataset in DATASET_NAMES}
    top5_correct = 0
    total = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        dataset_ids = batch["dataset_id"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        features = model.encode(images)
        for dataset_id in dataset_ids.unique(sorted=True).tolist():
            mask = dataset_ids == dataset_id
            dataset = DATASET_NAMES[int(dataset_id)]
            logits = model.logits_for_dataset(features[mask], dataset_id)
            prob = logits.softmax(dim=1)
            truth = labels[mask]
            pred = logits.argmax(dim=1)
            k = min(5, logits.shape[1])
            top5_correct += int((logits.topk(k, dim=1).indices == truth[:, None]).any(dim=1).sum().item())
            for target, prediction in zip(truth.cpu().tolist(), pred.cpu().tolist()):
                confusions[dataset][target, prediction] += 1
            targets[dataset].append(truth.cpu().numpy())
            scores[dataset].append(prob.cpu().numpy())
            total += int(truth.numel())
    metrics = metrics_from_confusions(confusions)
    metrics["top5_accuracy"] = top5_correct / max(total, 1)
    metrics.update(probability_metrics(targets, scores))
    return metrics, confusions, targets, scores


def main() -> None:
    args = parse_args()
    for checkpoint in args.checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    client_rows = []
    combined_confusions = empty_confusions()
    combined_targets = {dataset: [] for dataset in DATASET_NAMES}
    combined_scores = {dataset: [] for dataset in DATASET_NAMES}

    for client_id, checkpoint in enumerate(args.checkpoints):
        seed_everything(args.seed + client_id * 10000 + 200000)
        model = MultiDatasetRTDETRClassifier(
            dropout=0.1,
            moe_enabled=args.moe_enabled,
            moe_num_experts=args.moe_num_experts,
            moe_top_k=args.moe_top_k,
            moe_bottleneck=args.moe_bottleneck,
            moe_gamma_init=args.moe_gamma_init,
        )
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False), strict=True)
        model.to(device)
        config = {
            "classification_data_root": str(args.data_root),
            "classification_stratified_manifest": str(args.manifest),
            "classification_stratified_manifest_sha256": args.manifest_sha256,
            "classification_image_size": args.image_size,
            "classification_batch_size": args.batch_size,
            "classification_eval_batch_size": args.batch_size,
            "classification_master_seed": args.seed,
            "classification_num_workers": args.num_workers,
            "classification_train_max_samples": 0,
            "classification_eval_max_samples": 0,
            "classification_pcam_group_balanced": False,
            "classification_group_audit_unavailable": os.getenv(
                "CLASSIFICATION_GROUP_AUDIT_UNAVAILABLE", "0"
            ) == "1",
        }
        loader, stats = build_stratified_loader(config, client_id, "test", 200000)
        metrics, confusions, targets, scores = evaluate_once(model, loader, device)
        for dataset in DATASET_NAMES:
            combined_confusions[dataset] += confusions[dataset]
            combined_targets[dataset].extend(targets[dataset])
            combined_scores[dataset].extend(scores[dataset])
        row = {
            "client_id": client_id,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "test": stats,
            "metrics": metrics,
        }
        client_rows.append(row)
        (args.output_dir / f"client{client_id}_test_metrics.json").write_text(
            json.dumps(row, indent=2, sort_keys=True)
        )
        print(json.dumps({
            "client": client_id,
            "precision": metrics["weighted_precision"],
            "recall": metrics["weighted_recall"],
            "f1": metrics["weighted_f1"],
            "weighted_auroc": metrics["weighted_auroc"],
            "weighted_auprc": metrics["weighted_auprc"],
        }, sort_keys=True), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    combined = metrics_from_confusions(combined_confusions)
    combined.update(probability_metrics(combined_targets, combined_scores))
    keys = ["weighted_precision", "weighted_recall", "weighted_f1", "weighted_auroc", "weighted_auprc"]
    mean_sd = {}
    test_counts = np.asarray(
        [row["metrics"]["eval_examples"] for row in client_rows], dtype=np.float64
    )
    if np.any(test_counts <= 0) or float(test_counts.sum()) <= 0:
        raise RuntimeError(f"Invalid per-client test counts: {test_counts.tolist()}")
    sample_weighted = {}
    for key in keys:
        values = np.asarray([row["metrics"][key] for row in client_rows], dtype=np.float64)
        mean_sd[key] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)), "n_clients": 5}
        sample_weighted[key] = float(np.average(values, weights=test_counts))
    report = {
        "protocol": "one_time_fixed_test_after_validation_only_client_specific_selection",
        "selection_description": args.selection_description,
        "test_used_for_selection": False,
        "seed": args.seed,
        "data_root": str(args.data_root),
        "manifest": str(args.manifest),
        "manifest_sha256": args.manifest_sha256,
        "source_manifest_sha256": args.source_manifest_sha256,
        "pcam_group_balanced": False,
        "moe_enabled": args.moe_enabled,
        "moe_num_experts": args.moe_num_experts,
        "moe_top_k": args.moe_top_k,
        "moe_bottleneck": args.moe_bottleneck,
        "moe_gamma_init": args.moe_gamma_init,
        "clients": client_rows,
        "client_mean_sample_sd": mean_sd,
        "client_sample_count_weighted_metrics": {
            "definition": "sum(client_metric * client_test_n) / sum(client_test_n)",
            "total_test_n": int(test_counts.sum()),
            "client_test_n": [int(value) for value in test_counts],
            **sample_weighted,
        },
        "combined_test_metrics": combined,
    }
    (args.output_dir / "one_time_test_evaluation_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    print(json.dumps({"client_mean_sample_sd": mean_sd,
                      "client_sample_count_weighted_metrics": sample_weighted,
                      "combined": {key: combined[key] for key in keys}},
                     indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
