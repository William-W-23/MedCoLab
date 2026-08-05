from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from flwr.common.logger import configure
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_convert
from ultralytics.models.utils.loss import RTDETRDetectionLoss

from models import RTDETR_L, RTDETR_L_WithASEM
FINETUNE_MODEL_REGISTRY = {
    "RTDETR_L": RTDETR_L,
    "RTDETR_L_WithASEM": RTDETR_L_WithASEM,
}

def build_finetune_model(model_variant: str, nc: int = 20):
    if model_variant not in FINETUNE_MODEL_REGISTRY:
        raise ValueError(f"Unsupported finetune_model_variant: {model_variant}")
    return FINETUNE_MODEL_REGISTRY[model_variant](nc=nc)

from .voc_utils import (
    build_dirichlet_partitions,
    build_iid_partitions,
    discover_yolo_split_records,
    sample_client_indices,
    summarize_partition,
)

configure(identifier="finetune_task")

FINETUNE_DEFAULTS = {
    "finetune_init": "moco",
    "finetune_model_variant": "RTDETR_L",
    "finetune_ssl_ckpt_path": "outputs/ssl_moco_RTDETR_L_FedAvg_voc2007_5clients/ssl_global_backbone.pt",
    "finetune_dataset": "voc2007",
    "finetune_dataset_root": "datasets/VOC2007",
    "finetune_num_clients": 5,
    "finetune_partition_method": "dirichlet",
    "finetune_dirichlet_alpha": 0.5,
    "finetune_label_fraction": 0.05,
    "finetune_batch_size": 4,
    "finetune_train_max_batches": 0,
    "finetune_eval_max_batches": 0,
    "finetune_fraction_evaluate": 1.0,
    "finetune_freeze_backbone_epochs": 1,
    "finetune_unfreeze_lr": 1e-5,
    "finetune_detector_ckpt": "weights/rtdetr-l.pt",
    "finetune_domain_id": -1,
}

LOSS_KWARGS = {
    "use_vfl": True,
    "use_eqlv2": True,
    "loss_gain": {
        "class": 0.5,
        "eqlv2": 0.5,
        "bbox": 5.0,
        "giou": 2.0,
        "no_object": 0.1,
        "mask": 1.0,
        "dice": 1.0,
    },
    "gamma": 1.5,
    "alpha": 0.25,
    "eql_gamma": 12.0,
    "eql_mu": 0.8,
    "eql_alpha": 4.0,
}


def normalize_finetune_init(value: str) -> str:
    aliases = {
        "from_ssl_backbone": "moco",
        "ssl": "moco",
        "moco_backbone": "moco",
        "supervised_detector": "supervised",
        "detector": "supervised",
        "none": "random",
    }
    normalized = aliases.get(str(value), str(value))
    if normalized not in {"random", "supervised", "moco"}:
        raise ValueError(f"Unsupported finetune_init: {value}")
    return normalized


def get_finetune_config(run_config) -> Dict:
    config = dict(FINETUNE_DEFAULTS)
    if run_config is not None:
        for key in FINETUNE_DEFAULTS:
            if key in run_config:
                config[key] = run_config[key]
    config["finetune_init"] = normalize_finetune_init(config["finetune_init"])
    return config


def get_finetune_output_dir(config: Dict) -> Path:
    output_dir = Path("outputs") / (
        f"finetune_{config['finetune_init']}_{config['finetune_model_variant']}_{config['finetune_dataset']}_"
        f"{config['finetune_num_clients']}clients_{int(float(config['finetune_label_fraction']) * 100)}pct"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_finetune_mode_signature(config: Dict) -> str:
    return "|".join(
        [
            f"task:finetune",
            f"init:{config['finetune_init']}",
            f"model:{config['finetune_model_variant']}",
            f"dataset:{config['finetune_dataset']}",
            f"clients:{config['finetune_num_clients']}",
            f"label_fraction:{config['finetune_label_fraction']}",
        ]
    )


class DetectionDataset(Dataset):
    def __init__(self, records: List[Dict], train: bool, domain_id: int = -1):
        self.records = records
        self.domain_id = int(domain_id)
        self.transform = self._build_transform(train)

    @staticmethod
    def _build_transform(train: bool):
        ops = []
        if train:
            ops.extend(
                [
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(p=0.3),
                ]
            )
        ops.extend(
            [
                A.LongestMaxSize(max_size=640),
                A.PadIfNeeded(min_height=640, min_width=640, border_mode=0, fill=(114, 114, 114)),
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
                ToTensorV2(),
            ]
        )
        return A.Compose(ops, bbox_params=A.BboxParams(format="yolo", min_visibility=0.0))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = np.array(Image.open(record["image_path"]).convert("RGB"))
        raw_labels = record["labels"]
        bboxes = [[item[1], item[2], item[3], item[4], item[0]] for item in raw_labels]
        transformed = self.transform(image=image, bboxes=bboxes)
        final_labels = []
        for bbox in transformed["bboxes"]:
            x, y, w, h, cls = bbox
            final_labels.append([int(cls), float(x), float(y), float(w), float(h)])
        return {"pixel_values": transformed["image"], "labels": final_labels, "domain_id": self.domain_id}


def rtdetr_collate_fn(batch):
    pixel_values = []
    batch_bboxes = []
    batch_cls = []
    gt_groups = []
    batch_idx_list = []
    domain_id_list = []

    for i, item in enumerate(batch):
        pixel_values.append(item["pixel_values"])
        labels = item["labels"]
        if "domain_id" in item:
            domain_id_list.append(int(item["domain_id"]))
        num_gt = len(labels)
        gt_groups.append(num_gt)
        if num_gt == 0:
            continue
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        batch_cls.append(labels_tensor[:, 0].long())
        batch_bboxes.append(labels_tensor[:, 1:])
        batch_idx_list.append(torch.full((num_gt,), i, dtype=torch.long))

    images = torch.stack(pixel_values, dim=0)
    if batch_cls:
        cls = torch.cat(batch_cls, dim=0)
        bboxes = torch.cat(batch_bboxes, dim=0)
        batch_idx = torch.cat(batch_idx_list, dim=0)
    else:
        cls = torch.zeros(0, dtype=torch.long)
        bboxes = torch.zeros(0, 4, dtype=torch.float32)
        batch_idx = torch.zeros(0, dtype=torch.long)

    result = {
        "images": images,
        "cls": cls,
        "bboxes": bboxes,
        "gt_groups": gt_groups,
        "batch_idx": batch_idx,
    }
    if len(domain_id_list) == len(batch):
        result["domain_id"] = torch.tensor(domain_id_list, dtype=torch.long)
    return result


def build_detection_loss(device: torch.device):
    return RTDETRDetectionLoss(nc=20, **LOSS_KWARGS).to(device)


def extract_state_dict_from_checkpoint(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "ema" in checkpoint and checkpoint["ema"] is not None:
        source_model = checkpoint["ema"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        source_model = checkpoint["model"]
    else:
        source_model = checkpoint
    return source_model.state_dict() if hasattr(source_model, "state_dict") else source_model


def load_detector_weights(model: RTDETR_L, checkpoint_path: str) -> int:
    source_state = extract_state_dict_from_checkpoint(checkpoint_path)
    target_state = model.state_dict()
    filtered = {
        key: value
        for key, value in source_state.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    model.load_state_dict(filtered, strict=False)
    return len(filtered)


def load_ssl_backbone_weights(model: RTDETR_L, ssl_ckpt_path: str) -> int:
    checkpoint = torch.load(ssl_ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "encoder_q_backbone" in checkpoint:
        state_dict = checkpoint["encoder_q_backbone"]
    else:
        state_dict = checkpoint

    target_state = model.state_dict()
    filtered = {}
    for key, value in state_dict.items():
        stripped_key = key
        if stripped_key.startswith("encoder_q."):
            stripped_key = stripped_key.replace("encoder_q.", "", 1)
        if stripped_key.startswith("detector."):
            stripped_key = stripped_key.replace("detector.", "", 1)
        if not stripped_key.startswith("model."):
            continue
        parts = stripped_key.split(".")
        if len(parts) < 3:
            continue
        try:
            layer_idx = int(parts[1])
        except ValueError:
            continue
        if layer_idx > 9:
            continue
        if stripped_key in target_state and target_state[stripped_key].shape == value.shape:
            filtered[stripped_key] = value

    model.load_state_dict(filtered, strict=False)
    return len(filtered)


def set_backbone_trainable(model: RTDETR_L, trainable: bool) -> None:
    for idx in range(10):
        for param in model.model[idx].parameters():
            param.requires_grad = trainable


def build_optimizer(model: RTDETR_L, lr: float, backbone_lr: float):
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("model.0") or name.startswith("model.1") or name.startswith("model.2") or name.startswith("model.3") or name.startswith("model.4") or name.startswith("model.5") or name.startswith("model.6") or name.startswith("model.7") or name.startswith("model.8") or name.startswith("model.9"):
            backbone_params.append(param)
        else:
            other_params.append(param)
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if other_params:
        param_groups.append({"params": other_params, "lr": lr})
    return torch.optim.AdamW(param_groups, weight_decay=1e-4)


def train_finetune(
    model: RTDETR_L,
    trainloader: DataLoader,
    device: torch.device,
    local_epochs: int,
    lr: float,
    unfreeze_lr: float,
    freeze_backbone_epochs: int,
    max_batches: int = 0,
) -> Dict[str, float]:
    criterion = build_detection_loss(device)
    total_loss = 0.0
    total_batches = 0
    optimizer = None
    backbone_frozen = None

    model.to(device)
    model.train()

    for epoch in range(local_epochs):
        should_freeze = epoch < freeze_backbone_epochs
        if backbone_frozen is None or backbone_frozen != should_freeze:
            backbone_frozen = should_freeze
            set_backbone_trainable(model, not should_freeze)
            optimizer = build_optimizer(model, lr=lr, backbone_lr=unfreeze_lr)

        for batch in trainloader:
            images = batch["images"].to(device)
            batch["cls"] = batch["cls"].to(device)
            batch["bboxes"] = batch["bboxes"].to(device)
            batch["batch_idx"] = batch["batch_idx"].to(device)
            if "domain_id" in batch and torch.is_tensor(batch["domain_id"]):
                batch["domain_id"] = batch["domain_id"].to(device)

            optimizer.zero_grad()
            outputs = model(images, batch=batch)
            dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = outputs
            if enc_bboxes is not None:
                enc_bboxes = enc_bboxes.unsqueeze(0)
                enc_scores = enc_scores.unsqueeze(0)
            loss_dict = criterion(
                preds=(dec_bboxes, dec_scores),
                batch=batch,
                dn_bboxes=enc_bboxes,
                dn_scores=enc_scores,
                dn_meta=dn_meta,
            )
            loss = sum(loss_dict.values())
            aux_loss = model.get_aux_loss() if hasattr(model, "get_aux_loss") else None
            if aux_loss is not None:
                loss = loss + aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

            if max_batches > 0 and total_batches >= max_batches:
                break

        if max_batches > 0 and total_batches >= max_batches:
            break

    return {
        "train_loss": total_loss / max(total_batches, 1),
        "backbone_frozen_epochs": float(min(local_epochs, freeze_backbone_epochs)),
    }


def safe_detection_metric(metrics_dict: Dict[str, torch.Tensor], key: str) -> float:
    value = metrics_dict.get(key)
    if value is None:
        return 0.0
    number = float(value.item())
    if number != number or number < 0:
        return 0.0
    return number


def evaluate_detection(
    model: RTDETR_L,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> Dict[str, float]:
    criterion = build_detection_loss(device)
    metric = MeanAveragePrecision(box_format="cxcywh", iou_type="bbox").to(device)
    total_loss = 0.0
    total_batches = 0

    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            batch["cls"] = batch["cls"].to(device)
            batch["bboxes"] = batch["bboxes"].to(device)
            if "domain_id" in batch and torch.is_tensor(batch["domain_id"]):
                batch["domain_id"] = batch["domain_id"].to(device)
            outputs = model(images, batch=batch)
            if not (isinstance(outputs, tuple) and len(outputs) == 2):
                continue
            inference_out, raw_out = outputs
            dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = raw_out
            if enc_bboxes is not None:
                enc_bboxes = enc_bboxes.unsqueeze(0)
                enc_scores = enc_scores.unsqueeze(0)
            loss_dict = criterion(
                preds=(dec_bboxes, dec_scores),
                batch=batch,
                dn_bboxes=enc_bboxes,
                dn_scores=enc_scores,
                dn_meta=dn_meta,
            )
            total_loss += float(sum(loss_dict.values()).item())
            total_batches += 1

            target_list = []
            current_idx = 0
            for num_gt in batch["gt_groups"]:
                if num_gt > 0:
                    target_list.append(
                        {
                            "boxes": batch["bboxes"][current_idx : current_idx + num_gt],
                            "labels": batch["cls"][current_idx : current_idx + num_gt],
                        }
                    )
                    current_idx += num_gt
                else:
                    target_list.append(
                        {
                            "boxes": torch.empty(0, 4, device=device),
                            "labels": torch.empty(0, dtype=torch.long, device=device),
                        }
                    )
            pred_list = []
            for pred_item in inference_out:
                boxes = pred_item[:, :4]
                class_scores = pred_item[:, 4:]
                scores, labels = class_scores.max(dim=-1)
                pred_list.append({"boxes": boxes, "scores": scores, "labels": labels})
            metric.update(pred_list, target_list)

            if max_batches > 0 and total_batches >= max_batches:
                break

    metrics_dict = metric.compute()
    return {
        "eval_loss": total_loss / max(total_batches, 1),
        "eval_map": safe_detection_metric(metrics_dict, "map"),
        "eval_map50": safe_detection_metric(metrics_dict, "map_50"),
        "eval_map75": safe_detection_metric(metrics_dict, "map_75"),
        "eval_recall": safe_detection_metric(metrics_dict, "mar_100"),
        "eval_batches": float(total_batches),
    }


def build_finetune_dataloaders(partition_id: int, config: Dict):
    dataset_root = config["finetune_dataset_root"]
    train_records = discover_yolo_split_records(dataset_root, "train")
    val_records = discover_yolo_split_records(dataset_root, "val")
    test_records = discover_yolo_split_records(dataset_root, "test")

    train_partitions = build_dirichlet_partitions(
        train_records,
        num_clients=int(config["finetune_num_clients"]),
        alpha=float(config["finetune_dirichlet_alpha"]),
        seed=42,
    )
    val_partitions = build_iid_partitions(len(val_records), int(config["finetune_num_clients"]), seed=42)
    test_partitions = build_iid_partitions(len(test_records), int(config["finetune_num_clients"]), seed=84)

    train_indices = sample_client_indices(
        train_partitions[int(partition_id)],
        fraction=float(config["finetune_label_fraction"]),
        seed=100 + int(partition_id),
    )
    val_indices = val_partitions[int(partition_id)]
    test_indices = test_partitions[int(partition_id)]

    domain_id = int(config.get("finetune_domain_id", -1))
    train_dataset = DetectionDataset([train_records[idx] for idx in train_indices], train=True, domain_id=domain_id)
    val_dataset = DetectionDataset([val_records[idx] for idx in val_indices], train=False, domain_id=domain_id)
    test_dataset = DetectionDataset([test_records[idx] for idx in test_indices], train=False, domain_id=domain_id)

    batch_size = int(config["finetune_batch_size"])
    loaders = (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=rtdetr_collate_fn),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=rtdetr_collate_fn),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=rtdetr_collate_fn),
    )
    stats = {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "train_hist": summarize_partition(train_records, train_indices),
    }
    return loaders, stats


def save_finetune_metadata(output_dir: Path, config: Dict, extra: Dict) -> None:
    with (output_dir / "finetune_meta.json").open("w", encoding="utf-8") as handle:
        json.dump({"config": config, **extra}, handle, indent=2)
