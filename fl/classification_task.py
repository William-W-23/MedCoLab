from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.classification_moe import (
    DATASET_CLASSES,
    DATASET_NAMES,
    DATASET_PROFILE,
    DATASET_PROFILES,
    DATASET_TO_ID,
    MultiDatasetRTDETRClassifier,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _worker_seed(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class Medical5ClassificationDataset(Dataset):
    def __init__(self, client_root: str | Path, split: str, image_size: int, train: bool):
        self.client_root = Path(client_root)
        self.split = split
        image_root = self.client_root / "labeled" / "images" / split
        if not image_root.is_dir():
            raise FileNotFoundError(f"Missing labeled split: {image_root}")

        local_maps = {
            dataset: {label: idx for idx, label in enumerate(labels)}
            for dataset, labels in DATASET_CLASSES.items()
        }
        self.records = []
        for class_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
            if "__" not in class_dir.name:
                raise ValueError(f"Expected dataset__class directory, got {class_dir}")
            dataset, label = class_dir.name.split("__", 1)
            if dataset not in local_maps or label not in local_maps[dataset]:
                raise ValueError(f"Unknown class directory: {class_dir.name}")
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.records.append(
                        {
                            "path": str(image_path),
                            "dataset": dataset,
                            "dataset_id": DATASET_TO_ID[dataset],
                            "label": local_maps[dataset][label],
                            "class_name": class_dir.name,
                        }
                    )
        if not self.records:
            raise FileNotFoundError(f"No images found under {image_root}")

        ops = []
        if train:
            ops.extend(
                [
                    transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0)),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        else:
            ops.append(transforms.Resize((image_size, image_size)))
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self.transform = transforms.Compose(ops)
        # PCam labels are defined by tissue at the centre of a fixed patch. The
        # generic RandomResizedCrop can remove that region and create label noise,
        # so PCam keeps the full field of view while receiving mild stain-robust
        # perturbations.
        if train:
            self.pcam_transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08, hue=0.02)],
                        p=0.5,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ]
            )
        else:
            self.pcam_transform = self.transform
        self.class_hist = Counter(record["class_name"] for record in self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record["path"]) as image:
            transform = self.pcam_transform if record["dataset"] == "pcam" else self.transform
            tensor = transform(image.convert("RGB"))
        return {
            "image": tensor,
            "dataset_id": torch.tensor(record["dataset_id"], dtype=torch.long),
            "label": torch.tensor(record["label"], dtype=torch.long),
            "image_path": record["path"],
        }


def build_dataloader(
    data_root: str | Path,
    partition_id: int,
    split: str,
    image_size: int,
    batch_size: int,
    master_seed: int,
    round_num: int,
    num_workers: int,
    max_samples: int = 0,
):
    client_root = Path(data_root) / f"client{partition_id}"
    train = split == "train"
    dataset = Medical5ClassificationDataset(client_root, split, image_size, train=train)
    if max_samples > 0 and max_samples < len(dataset.records):
        # Smoke mode: deterministic round-robin sampling across every available class
        # so all dataset-specific heads are exercised instead of taking one sorted prefix.
        by_class = {}
        for record in dataset.records:
            by_class.setdefault(record["class_name"], []).append(record)
        selected = []
        offset = 0
        class_names = sorted(by_class)
        while len(selected) < max_samples:
            added = False
            for class_name in class_names:
                records = by_class[class_name]
                if offset < len(records):
                    selected.append(records[offset])
                    added = True
                    if len(selected) == max_samples:
                        break
            if not added:
                break
            offset += 1
        dataset.records = selected
        dataset.class_hist = Counter(record["class_name"] for record in dataset.records)
    split_offset = {"train": 0, "val": 100_000, "test": 200_000}[split]
    loader_seed = master_seed + partition_id * 10_000 + round_num + split_offset
    generator = torch.Generator().manual_seed(loader_seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_worker_seed,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
    return loader, {
        "client": partition_id,
        "split": split,
        "num_samples": len(dataset),
        "class_hist": dict(sorted(dataset.class_hist.items())),
        "root": str(client_root),
    }


def load_round0_backbone(model: MultiDatasetRTDETRClassifier, checkpoint_path: str | Path):
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    target = model.state_dict()
    filtered = {}
    for key, value in source.items():
        if not key.startswith("model."):
            continue
        mapped = "backbone." + key[len("model.") :]
        if mapped in target and tuple(target[mapped].shape) == tuple(value.shape):
            filtered[mapped] = value
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return {
        "loaded": len(filtered),
        "source": len(source),
        "target": len(target),
        "missing": len(missing),
        "unexpected": len(unexpected),
    }


def _class_weights(class_hist: Dict[str, int], dataset: str, device: torch.device, power: float = 1.0):
    counts = torch.tensor(
        [float(class_hist.get(f"{dataset}__{label}", 0)) for label in DATASET_CLASSES[dataset]],
        dtype=torch.float32,
        device=device,
    )
    weights = counts.sum() / counts.clamp_min(1.0)
    weights = weights.clamp_min(1e-12).pow(float(power))
    return weights / weights.mean().clamp_min(1e-12)


def train_one_round(
    model: MultiDatasetRTDETRClassifier,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    max_batches: int = 0,
    backbone_lr: float | None = None,
    head_lr: float | None = None,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.0,
    class_weight_power: float = 1.0,
    moe_lr: float | None = None,
    moe_balance_loss_weight: float = 0.01,
):
    model.train()
    if backbone_lr is not None or head_lr is not None:
        parameter_groups = [
            {"params": model.backbone.parameters(), "lr": float(backbone_lr if backbone_lr is not None else lr)},
            {"params": model.heads.parameters(), "lr": float(head_lr if head_lr is not None else lr)},
        ]
        if model.moe is not None:
            parameter_groups.append({"params": model.moe.parameters(), "lr": float(moe_lr if moe_lr is not None else lr)})
        optimizer = torch.optim.AdamW(parameter_groups, weight_decay=float(weight_decay))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(weight_decay))
    weights = {
        dataset: _class_weights(loader.dataset.class_hist, dataset, device, power=class_weight_power)
        for dataset in DATASET_NAMES
    }
    total_loss = 0.0
    total_seen = 0
    total_correct = 0
    total_batches = 0
    stop = False
    for _ in range(epochs):
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            dataset_ids = batch["dataset_id"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            features = model.encode(images)
            loss_sum = features.new_zeros(())
            batch_correct = 0
            for dataset_id in dataset_ids.unique(sorted=True).tolist():
                mask = dataset_ids == dataset_id
                dataset = DATASET_NAMES[int(dataset_id)]
                logits = model.logits_for_dataset(features[mask], dataset_id)
                loss_sum = loss_sum + F.cross_entropy(
                    logits,
                    labels[mask],
                    weight=weights[dataset],
                    reduction="sum",
                    label_smoothing=float(label_smoothing),
                )
                batch_correct += int((logits.argmax(dim=1) == labels[mask]).sum().item())
            loss = loss_sum / max(int(labels.numel()), 1)
            if model.moe is not None:
                loss = loss + float(moe_balance_loss_weight) * model.moe.load_balance_loss()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())
            total_seen += int(labels.numel())
            total_correct += batch_correct
            total_batches += 1
            if max_batches > 0 and total_batches >= max_batches:
                stop = True
                break
        if stop:
            break
    return {
        "train_loss": total_loss / max(total_batches, 1),
        "train_accuracy": total_correct / max(total_seen, 1),
        "train_batches": float(total_batches),
        "trained_examples": float(total_seen),
    }


def empty_confusions():
    return {
        dataset: torch.zeros((len(classes), len(classes)), dtype=torch.long)
        for dataset, classes in DATASET_CLASSES.items()
    }


@torch.no_grad()
def evaluate(
    model: MultiDatasetRTDETRClassifier,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
):
    model.eval()
    confusions = empty_confusions()
    total_loss = 0.0
    total_batches = 0
    total_seen = 0
    top5_correct = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        dataset_ids = batch["dataset_id"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        features = model.encode(images)
        loss_sum = features.new_zeros(())
        for dataset_id in dataset_ids.unique(sorted=True).tolist():
            mask = dataset_ids == dataset_id
            dataset = DATASET_NAMES[int(dataset_id)]
            logits = model.logits_for_dataset(features[mask], dataset_id)
            targets = labels[mask]
            loss_sum += F.cross_entropy(logits, targets, reduction="sum")
            k = min(5, logits.shape[1])
            top5_correct += int((logits.topk(k, dim=1).indices == targets[:, None]).any(dim=1).sum().item())
            preds = logits.argmax(dim=1)
            cm = confusions[dataset]
            for target, pred in zip(targets.cpu().tolist(), preds.cpu().tolist()):
                cm[target, pred] += 1
        total_loss += float((loss_sum / max(int(labels.numel()), 1)).item())
        total_seen += int(labels.numel())
        total_batches += 1
        if max_batches > 0 and total_batches >= max_batches:
            break
    metrics = metrics_from_confusions(confusions)
    metrics.update(
        {
            "val_loss": total_loss / max(total_batches, 1),
            "top5_accuracy": top5_correct / max(total_seen, 1),
            "eval_batches": float(total_batches),
            "eval_examples": float(total_seen),
        }
    )
    metrics.update(flatten_confusions(confusions))
    return metrics


def flatten_confusions(confusions: Dict[str, torch.Tensor]):
    flat = {}
    for dataset, cm in confusions.items():
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                flat[f"cm__{dataset}__{row}__{col}"] = float(cm[row, col].item())
    return flat


def confusions_from_metrics(metric_records: Iterable[dict]):
    confusions = empty_confusions()
    for metrics in metric_records:
        for dataset, cm in confusions.items():
            for row in range(cm.shape[0]):
                for col in range(cm.shape[1]):
                    cm[row, col] += int(round(float(metrics.get(f"cm__{dataset}__{row}__{col}", 0))))
    return confusions


def metrics_from_confusions(confusions: Dict[str, torch.Tensor]):
    per_class = []
    total_correct = 0.0
    total_examples = 0.0
    result = {}
    for dataset, cm_tensor in confusions.items():
        cm = cm_tensor.double()
        support = cm.sum(dim=1)
        predicted = cm.sum(dim=0)
        tp = cm.diag()
        precision = tp / predicted.clamp_min(1.0)
        recall = tp / support.clamp_min(1.0)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
        dataset_total = float(cm.sum().item())
        dataset_correct = float(tp.sum().item())
        result[f"{dataset}_accuracy"] = dataset_correct / max(dataset_total, 1.0)
        result[f"{dataset}_macro_f1"] = float(f1.mean().item())
        for idx, label in enumerate(DATASET_CLASSES[dataset]):
            prefix = f"class__{dataset}__{label}"
            result[f"{prefix}__precision"] = float(precision[idx].item())
            result[f"{prefix}__recall"] = float(recall[idx].item())
            result[f"{prefix}__f1"] = float(f1[idx].item())
            result[f"{prefix}__support"] = float(support[idx].item())
            # For one-vs-rest class reporting, class accuracy is sensitivity.
            result[f"{prefix}__accuracy"] = float(recall[idx].item())
            result[f"{prefix}__error_count"] = float((support[idx] - tp[idx]).item())
            per_class.append((precision[idx], recall[idx], f1[idx], support[idx]))
        total_correct += dataset_correct
        total_examples += dataset_total

    precision = torch.stack([row[0] for row in per_class])
    recall = torch.stack([row[1] for row in per_class])
    f1 = torch.stack([row[2] for row in per_class])
    support = torch.stack([row[3] for row in per_class])
    support_total = support.sum().clamp_min(1.0)
    result.update(
        {
            "accuracy": total_correct / max(total_examples, 1.0),
            "macro_precision": float(precision.mean().item()),
            "macro_recall": float(recall.mean().item()),
            "macro_f1": float(f1.mean().item()),
            "weighted_precision": float((precision * support).sum().item() / support_total.item()),
            "weighted_recall": float((recall * support).sum().item() / support_total.item()),
            "weighted_f1": float((f1 * support).sum().item() / support_total.item()),
            "balanced_accuracy": float(recall.mean().item()),
            "eval_examples": float(total_examples),
        }
    )
    return result


def save_confusion_artifacts(confusions: Dict[str, torch.Tensor], output_dir: str | Path, prefix: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {dataset: cm.tolist() for dataset, cm in confusions.items()}
    (output / f"{prefix}_confusion_matrices.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    for dataset, cm in confusions.items():
        fig, ax = plt.subplots(figsize=(max(5, len(DATASET_CLASSES[dataset]) * 0.8), 5))
        image = ax.imshow(cm.numpy(), cmap="Blues")
        ax.set_title(f"{prefix}: {dataset}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(DATASET_CLASSES[dataset])), DATASET_CLASSES[dataset], rotation=45, ha="right")
        ax.set_yticks(range(len(DATASET_CLASSES[dataset])), DATASET_CLASSES[dataset])
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(output / f"{prefix}_confusion_{dataset}.png", dpi=160)
        plt.close(fig)
