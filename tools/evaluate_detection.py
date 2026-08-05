#!/usr/bin/env python3
"""Leakage-safe audit of personalized detection checkpoints.

Each client process performs inference on official validation and test splits once,
then evaluates max-detections 100 and 300. Thresholds are selected from validation
only. Aggregate mode combines the five client outputs and computes client mean/SD,
GT-weighted, and pooled TP/FP/FN summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("FL_CURRENT_DATASET", "medical5_detection_clientfirst_seed42")

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_convert, box_iou

from fl.detection_task import DATASET_CONFIGS, _Medical5YoloDataset, rtdetr_collate_fn
from models import RTDETR_L_WithASEM


DEFAULT_DATASET_KEY = "medical5_detection_clientfirst_seed42"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--client-id", type=int, choices=range(5))
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--score-floor", type=float, default=0.001)
    parser.add_argument("--recall-target", type=float, default=0.75)
    parser.add_argument("--dataset-key", default=DEFAULT_DATASET_KEY)
    return parser.parse_args()


def resolve_client_dir(run_dir: Path, client_id: int, client_name: str) -> Path:
    candidates = [
        run_dir / f"client{client_id}" / f"client{client_id}_{client_name}",
        run_dir / f"client{client_id}_{client_name}",
    ]
    for candidate in candidates:
        if (candidate / "best_model_by_map50.pt").is_file():
            return candidate
    raise FileNotFoundError(f"client{client_id} checkpoint not found under {run_dir}")


def load_split_records(
    model: torch.nn.Module,
    client_cfg: dict[str, Any],
    split: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[list[dict[str, torch.Tensor]], int]:
    dataset = _Medical5YoloDataset(
        client_cfg["data_dir"], split, client_cfg.get("domain_id", 0)
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=rtdetr_collate_fn,
        num_workers=num_workers,
    )
    records: list[dict[str, torch.Tensor]] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["images"].to(device)
            for key in ("cls", "bboxes", "batch_idx"):
                batch[key] = batch[key].to(device)
            if "domain_id" in batch and torch.is_tensor(batch["domain_id"]):
                batch["domain_id"] = batch["domain_id"].to(device)
            inference_out, _ = model(images, batch=batch)
            start = 0
            for image_index, num_gt_value in enumerate(batch["gt_groups"]):
                num_gt = int(num_gt_value)
                gt_boxes = batch["bboxes"][start : start + num_gt].detach().cpu()
                gt_labels = batch["cls"][start : start + num_gt].long().detach().cpu()
                start += num_gt
                item = inference_out[image_index].detach().cpu()
                score_matrix = item[:, 4:]
                scores, labels = score_matrix.max(dim=-1)
                records.append(
                    {
                        "boxes": item[:, :4].float(),
                        "scores": scores.float(),
                        "labels": labels.long(),
                        "gt_boxes": gt_boxes.float(),
                        "gt_labels": gt_labels,
                    }
                )
    return records, len(dataset)


def truncate_record(record: dict[str, torch.Tensor], max_det: int) -> dict[str, torch.Tensor]:
    order = torch.argsort(record["scores"], descending=True)[:max_det]
    return {
        "boxes": record["boxes"][order],
        "scores": record["scores"][order],
        "labels": record["labels"][order],
        "gt_boxes": record["gt_boxes"],
        "gt_labels": record["gt_labels"],
    }


def match_curves(
    records: list[dict[str, torch.Tensor]],
    max_det: int,
    nc: int,
    iou_threshold: float,
    score_floor: float,
) -> dict[str, np.ndarray]:
    scores_all: list[float] = []
    tp_all: list[int] = []
    class_all: list[int] = []
    iou_all: list[float] = []
    gt_counts = np.zeros(nc, dtype=np.int64)

    for raw in records:
        record = truncate_record(raw, max_det)
        for label in record["gt_labels"].tolist():
            gt_counts[int(label)] += 1
        for class_id in range(nc):
            pred_mask = (record["labels"] == class_id) & (record["scores"] >= score_floor)
            pred_boxes = record["boxes"][pred_mask]
            pred_scores = record["scores"][pred_mask]
            if pred_scores.numel() == 0:
                continue
            order = torch.argsort(pred_scores, descending=True)
            pred_boxes, pred_scores = pred_boxes[order], pred_scores[order]
            gt_boxes = record["gt_boxes"][record["gt_labels"] == class_id]
            matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
            ious = (
                box_iou(
                    box_convert(pred_boxes, "cxcywh", "xyxy"),
                    box_convert(gt_boxes, "cxcywh", "xyxy"),
                )
                if len(gt_boxes)
                else None
            )
            for pred_index, score in enumerate(pred_scores.tolist()):
                is_tp = 0
                matched_iou = 0.0
                if ious is not None:
                    iou_value, gt_index = ious[pred_index].max(0)
                    if float(iou_value) >= iou_threshold and not bool(matched[gt_index]):
                        matched[gt_index] = True
                        is_tp = 1
                        matched_iou = float(iou_value)
                scores_all.append(float(score))
                tp_all.append(is_tp)
                class_all.append(class_id)
                iou_all.append(matched_iou)

    return {
        "scores": np.asarray(scores_all, dtype=np.float32),
        "is_tp": np.asarray(tp_all, dtype=np.uint8),
        "class_id": np.asarray(class_all, dtype=np.int16),
        "matched_iou": np.asarray(iou_all, dtype=np.float32),
        "gt_counts": gt_counts,
        "num_images": np.asarray([len(records)], dtype=np.int64),
    }


def ap50_metrics(
    records: list[dict[str, torch.Tensor]], max_det: int, nc: int
) -> tuple[float, list[float | None]]:
    metric = MeanAveragePrecision(
        box_format="cxcywh",
        iou_type="bbox",
        iou_thresholds=[0.5],
        class_metrics=True,
        max_detection_thresholds=[1, 10, max_det],
    )
    for raw in records:
        record = truncate_record(raw, max_det)
        metric.update(
            [
                {
                    "boxes": record["boxes"],
                    "scores": record["scores"],
                    "labels": record["labels"],
                }
            ],
            [{"boxes": record["gt_boxes"], "labels": record["gt_labels"]}],
        )
    result = metric.compute()
    per_class: list[float | None] = [None] * nc
    for class_id, value in zip(result["classes"].tolist(), result["map_per_class"].tolist()):
        per_class[int(class_id)] = None if float(value) < 0 else float(value)
    return float(result["map"]), per_class


def best_threshold(curve: dict[str, np.ndarray], recall_target: float | None = None) -> dict[str, float]:
    scores = curve["scores"]
    is_tp = curve["is_tp"].astype(np.int64)
    total_gt = int(curve["gt_counts"].sum())
    if total_gt == 0 or scores.size == 0:
        return {"threshold": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    tp = np.cumsum(is_tp[order])
    fp = np.cumsum(1 - is_tp[order])
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / total_gt
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    valid = np.ones_like(f1, dtype=bool)
    if recall_target is not None:
        valid = recall >= recall_target
        if not valid.any():
            valid = np.ones_like(f1, dtype=bool)
    masked = np.where(valid, f1, -1.0)
    index = int(np.argmax(masked))
    return {
        "threshold": float(ordered_scores[index]),
        "precision": float(precision[index]),
        "recall": float(recall[index]),
        "f1": float(f1[index]),
    }


def operating_metrics(
    curve: dict[str, np.ndarray], threshold: float, nc: int
) -> dict[str, Any]:
    selected = curve["scores"] >= float(threshold)
    gt_counts = curve["gt_counts"].astype(np.int64)
    per_class = []
    total_tp = total_fp = 0
    for class_id in range(nc):
        class_mask = selected & (curve["class_id"] == class_id)
        tp = int(curve["is_tp"][class_mask].sum())
        fp = int(class_mask.sum()) - tp
        fn = int(gt_counts[class_id]) - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        per_class.append(
            {
                "class_id": class_id,
                "gt_boxes": int(gt_counts[class_id]),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
        total_tp += tp
        total_fp += fp
    total_gt = int(gt_counts.sum())
    total_fn = total_gt - total_tp
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_gt, 1)
    f1 = 2 * total_tp / max(2 * total_tp + total_fp + total_fn, 1)
    matched_ious = curve["matched_iou"][selected & (curve["is_tp"] == 1)]

    order = np.argsort(-curve["scores"], kind="stable")
    cumulative_tp = np.cumsum(curve["is_tp"][order].astype(np.int64))
    cumulative_fp = np.cumsum(1 - curve["is_tp"][order].astype(np.int64))
    fp_per_image = cumulative_fp / max(int(curve["num_images"][0]), 1)
    sensitivity = cumulative_tp / max(total_gt, 1)
    x = np.concatenate(([0.0], fp_per_image))
    y = np.concatenate(([0.0], sensitivity))
    unique_x, unique_indices = np.unique(x, return_index=True)
    unique_y = np.maximum.accumulate(y[unique_indices])
    dense_grid = np.linspace(0.0, 8.0, 801)
    dense_sensitivity = np.interp(dense_grid, unique_x, unique_y)
    froc_auc = float(np.trapz(dense_sensitivity, dense_grid) / 8.0)
    return {
        "threshold": float(threshold),
        "gt_boxes": total_gt,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "miou": float(matched_ious.mean()) if matched_ious.size else 0.0,
        "froc_auc": froc_auc,
        "per_class": per_class,
    }


def add_weighted_class_metrics(
    operating: dict[str, Any], ap50_per_class: list[float | None]
) -> None:
    supported = [
        row
        for row, ap50 in zip(operating["per_class"], ap50_per_class)
        if row["gt_boxes"] > 0 and ap50 is not None
    ]
    total_gt = sum(row["gt_boxes"] for row in supported)
    for row, ap50 in zip(operating["per_class"], ap50_per_class):
        row["ap50"] = ap50
    operating["class_gt_weighted"] = {
        "map50": sum(
            float(row["ap50"]) * row["gt_boxes"]
            for row in operating["per_class"]
            if row["gt_boxes"] > 0 and row["ap50"] is not None
        )
        / max(total_gt, 1),
        "precision": sum(row["precision"] * row["gt_boxes"] for row in supported)
        / max(total_gt, 1),
        "recall": sum(row["recall"] * row["gt_boxes"] for row in supported)
        / max(total_gt, 1),
        "f1": sum(row["f1"] * row["gt_boxes"] for row in supported)
        / max(total_gt, 1),
        "supported_gt_boxes": total_gt,
    }


def save_curve(path: Path, curve: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **curve)


def load_curve(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def evaluate_client(args: argparse.Namespace) -> None:
    if args.run_dir is None or args.client_id is None:
        raise SystemExit("--run-dir and --client-id are required for client evaluation")
    cfg = DATASET_CONFIGS[args.dataset_key]
    client_id = args.client_id
    client_cfg = cfg["clients"][client_id]
    client_dir = resolve_client_dir(args.run_dir, client_id, client_cfg["name"])
    model_path = client_dir / "best_model_by_map50.pt"
    meta_path = client_dir / "best_model_by_map50_meta.json"
    metrics_path = client_dir / "metrics.json"
    if not metrics_path.is_file():
        metrics_path = client_dir / "validation_metrics.json"
    meta = json.loads(meta_path.read_text())
    saved_metrics = json.loads(metrics_path.read_text())
    threshold_value = (
        saved_metrics["threshold"]
        if "threshold" in saved_metrics
        else saved_metrics["val_threshold"]
    )
    current_threshold = float(threshold_value)

    device = torch.device(args.device)
    model = RTDETR_L_WithASEM(nc=cfg["nc"]).to(device)
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()

    split_records = {}
    split_sizes = {}
    for split in ("val", "test"):
        split_records[split], split_sizes[split] = load_split_records(
            model, client_cfg, split, device, args.batch_size, args.num_workers
        )

    client_out = args.out_dir / f"client{client_id}"
    client_out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "client_id": client_id,
        "client_name": client_cfg["name"],
        "model_path": str(model_path),
        "model_meta": meta,
        "strict_checkpoint_load": True,
        "current_threshold_source": "saved selected_model_val",
        "current_threshold": current_threshold,
        "split_sizes": split_sizes,
        "test_data_used_for_selection": False,
        "max_detections": {},
    }
    for max_det in (100, 300):
        curves = {}
        ap = {}
        for split in ("val", "test"):
            curves[split] = match_curves(
                split_records[split], max_det, cfg["nc"], args.iou, args.score_floor
            )
            ap[split] = ap50_metrics(split_records[split], max_det, cfg["nc"])
            save_curve(client_out / f"{split}_maxdet{max_det}_curve.npz", curves[split])
        val_best = best_threshold(curves["val"])
        val_recall = best_threshold(curves["val"], args.recall_target)
        settings = {
            "saved_client_val_threshold": current_threshold,
            "recomputed_client_val_best_f1": val_best["threshold"],
            f"client_val_recall_at_least_{args.recall_target:g}": val_recall["threshold"],
        }
        max_result: dict[str, Any] = {
            "official_val_standard_map50": ap["val"][0],
            "official_test_standard_map50": ap["test"][0],
            "val_best_f1_threshold": val_best,
            "val_recall_constrained_threshold": val_recall,
            "threshold_settings": {},
        }
        for setting_name, threshold in settings.items():
            val_operating = operating_metrics(curves["val"], threshold, cfg["nc"])
            test_operating = operating_metrics(curves["test"], threshold, cfg["nc"])
            add_weighted_class_metrics(val_operating, ap["val"][1])
            add_weighted_class_metrics(test_operating, ap["test"][1])
            max_result["threshold_settings"][setting_name] = {
                "threshold_selected_on": "official_val",
                "threshold": threshold,
                "val": val_operating,
                "test": test_operating,
            }
        result["max_detections"][str(max_det)] = max_result

    (client_out / "audit.json").write_text(json.dumps(result, indent=2))
    print(
        f"client{client_id} complete: current threshold={current_threshold:.6f}; "
        f"maxdet100 test mAP50={result['max_detections']['100']['official_test_standard_map50']:.6f}; "
        f"maxdet300 test mAP50={result['max_detections']['300']['official_test_standard_map50']:.6f}",
        flush=True,
    )


def combine_curves(curves: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        "scores": np.concatenate([curve["scores"] for curve in curves]),
        "is_tp": np.concatenate([curve["is_tp"] for curve in curves]),
        "class_id": np.concatenate([curve["class_id"] for curve in curves]),
        "matched_iou": np.concatenate([curve["matched_iou"] for curve in curves]),
        "gt_counts": np.sum([curve["gt_counts"] for curve in curves], axis=0),
        "num_images": np.asarray(
            [sum(int(curve["num_images"][0]) for curve in curves)], dtype=np.int64
        ),
    }


def sample_sd(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=float), ddof=1)) if len(values) > 1 else 0.0


def weighted_mean_sd(values: list[float], weights: list[int]) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mean = float(np.average(x, weights=w))
    denominator = float(w.sum() - (w @ w) / w.sum())
    sd = math.sqrt(float((w * (x - mean) ** 2).sum()) / denominator) if denominator > 0 else 0.0
    return mean, sd


def aggregate(args: argparse.Namespace) -> None:
    cfg = DATASET_CONFIGS[args.dataset_key]
    client_payloads = [
        json.loads((args.out_dir / f"client{i}" / "audit.json").read_text())
        for i in range(5)
    ]
    summary: dict[str, Any] = {
        "protocol": "official-validation-only threshold selection; one locked test evaluation",
        "test_data_used_for_selection": False,
        "sd_definition": "sample SD across the five clients (ddof=1), not random-seed SD",
        "weighted_sd_definition": "unbiased reliability-weighted SD across five client estimates",
        "max_detections": {},
    }
    csv_rows = []
    for max_det in (100, 300):
        val_curves = [
            load_curve(args.out_dir / f"client{i}" / f"val_maxdet{max_det}_curve.npz")
            for i in range(5)
        ]
        test_curves = [
            load_curve(args.out_dir / f"client{i}" / f"test_maxdet{max_det}_curve.npz")
            for i in range(5)
        ]
        pooled_val = combine_curves(val_curves)
        pooled_test = combine_curves(test_curves)
        pooled_val_best = best_threshold(pooled_val)
        pooled_val_recall = best_threshold(pooled_val, args.recall_target)
        threshold_modes: dict[str, Any] = {}
        modes = {
            "saved_client_val_threshold": [
                float(payload["current_threshold"]) for payload in client_payloads
            ],
            "pooled_val_best_f1_unified_threshold": [pooled_val_best["threshold"]] * 5,
            f"pooled_val_recall_at_least_{args.recall_target:g}_unified_threshold": [
                pooled_val_recall["threshold"]
            ]
            * 5,
        }
        for mode_name, thresholds in modes.items():
            client_rows = []
            for client_id, threshold in enumerate(thresholds):
                operating = operating_metrics(test_curves[client_id], threshold, cfg["nc"])
                ap_per_class = client_payloads[client_id]["max_detections"][str(max_det)][
                    "threshold_settings"
                ]["saved_client_val_threshold"]["test"]["per_class"]
                ap_values = [row["ap50"] for row in ap_per_class]
                add_weighted_class_metrics(operating, ap_values)
                operating["standard_map50"] = client_payloads[client_id]["max_detections"][
                    str(max_det)
                ]["official_test_standard_map50"]
                client_rows.append(operating)
                csv_rows.append(
                    {
                        "max_detections": max_det,
                        "threshold_mode": mode_name,
                        "client_id": client_id,
                        "threshold": threshold,
                        "gt_boxes": operating["gt_boxes"],
                        "standard_map50": operating["standard_map50"],
                        "class_gt_weighted_map50": operating["class_gt_weighted"]["map50"],
                        "precision": operating["precision"],
                        "recall": operating["recall"],
                        "f1": operating["f1"],
                        "class_gt_weighted_f1": operating["class_gt_weighted"]["f1"],
                        "miou": operating["miou"],
                        "froc_auc": operating["froc_auc"],
                    }
                )
            weights = [row["gt_boxes"] for row in client_rows]
            fields = ["standard_map50", "precision", "recall", "f1", "miou", "froc_auc"]
            client_mean_sd = {
                field: {
                    "mean": float(np.mean([row[field] for row in client_rows])),
                    "sd": sample_sd([row[field] for row in client_rows]),
                }
                for field in fields
            }
            gt_weighted = {}
            for field in fields:
                mean, sd = weighted_mean_sd([row[field] for row in client_rows], weights)
                gt_weighted[field] = {"mean": mean, "sd": sd}
            class_weighted_fields = ["map50", "precision", "recall", "f1"]
            class_gt_weighted = {
                field: {
                    "mean": float(
                        np.mean([row["class_gt_weighted"][field] for row in client_rows])
                    ),
                    "sd": sample_sd(
                        [row["class_gt_weighted"][field] for row in client_rows]
                    ),
                }
                for field in class_weighted_fields
            }
            if len(set(thresholds)) == 1:
                pooled_operating = operating_metrics(
                    pooled_test, thresholds[0], cfg["nc"]
                )
            else:
                pooled_operating = {
                    "gt_boxes": sum(row["gt_boxes"] for row in client_rows),
                    "tp": sum(row["tp"] for row in client_rows),
                    "fp": sum(row["fp"] for row in client_rows),
                    "fn": sum(row["fn"] for row in client_rows),
                }
                tp, fp, fn = (
                    pooled_operating["tp"],
                    pooled_operating["fp"],
                    pooled_operating["fn"],
                )
                pooled_operating.update(
                    {
                        "precision": tp / max(tp + fp, 1),
                        "recall": tp / max(tp + fn, 1),
                        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                    }
                )
            threshold_modes[mode_name] = {
                "thresholds": thresholds,
                "client_results": client_rows,
                "client_mean_sd": client_mean_sd,
                "client_gt_weighted_mean_sd": gt_weighted,
                "within_client_class_gt_weighted_mean_sd": class_gt_weighted,
                "pooled_tp_fp_fn": pooled_operating,
            }
        summary["max_detections"][str(max_det)] = {
            "pooled_val_best_f1_threshold": pooled_val_best,
            "pooled_val_recall_constrained_threshold": pooled_val_recall,
            "threshold_modes": threshold_modes,
        }
    (args.out_dir / "protocol_audit_summary.json").write_text(json.dumps(summary, indent=2))
    with (args.out_dir / "protocol_audit_clients.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(args.out_dir / "protocol_audit_summary.json", flush=True)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        aggregate(args)
    else:
        evaluate_client(args)


if __name__ == "__main__":
    main()
