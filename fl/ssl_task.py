from __future__ import annotations

import copy
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from flwr.common import Array, ArrayRecord
from flwr.common.logger import configure, log
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

from models import RTDETR_L, RTDETR_L_WithASEM
from .voc_utils import build_dirichlet_partitions, discover_yolo_split_records, summarize_partition

SSL_MODEL_REGISTRY = {
    "RTDETR_L": RTDETR_L,
    "RTDETR_L_WithASEM": RTDETR_L_WithASEM,
}

def build_ssl_detector(model_variant: str, nc: int = 20) -> nn.Module:
    if model_variant not in SSL_MODEL_REGISTRY:
        raise ValueError(f"Unsupported ssl_model_variant: {model_variant}")
    return SSL_MODEL_REGISTRY[model_variant](nc=nc)

configure(identifier="ssl_task")

SSL_STATE_DIR = Path(os.environ.get("SSL_LOCAL_STATE_DIR", "/tmp/fl_ssl_local_state"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_dataloader_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
SSL_DEFAULTS = {
    "ssl_mode": "moco_rtdetr_backbone",
    "ssl_model_variant": "RTDETR_L",
    "ssl_pretrain_scope": "backbone_only",
    "ssl_dataset": "voc2007",
    "ssl_dataset_root": "datasets/VOC2007",
    "ssl_dataset_roots": "",
    "ssl_balance_datasets_per_batch": True,
    "ssl_num_clients": 5,
    "ssl_partition_method": "dirichlet",
    "ssl_dirichlet_alpha": 0.5,
    "ssl_feature_dim": 256,
    "ssl_proj_dim": 128,
    "ssl_queue_size": 4096,
    "ssl_momentum": 0.999,
    "ssl_temperature": 0.07,
    "ssl_local_epochs": 1,
    "ssl_batch_size": 8,
    "ssl_train_max_batches": 0,
    "ssl_lr": 1e-4,
    "ssl_image_size": 320,
    "ssl_fraction_train": 1.0,
    "ssl_enable_evaluate": False,
    "ssl_fraction_evaluate": 1.0,
    "ssl_eval_split": "val",
    "ssl_eval_max_batches": 20,
    "ssl_pretrained_detector_ckpt": "weights/rtdetr-l.pt",
    # --- Attentive Asymmetric Masking (enabled by default) ---
    "ssl_mask_enable": True,         # master switch; explicitly set False only for ablations
    "ssl_mask_mode": "foreground",   # "foreground" (mask high-response) | "background"
    "ssl_mask_ratio": 0.3,           # base fraction of feature-grid cells to mask on query view
    "ssl_mask_randomness": 0.5,      # 0 = always mask hardest cells, larger = more random
    # 小料1: curriculum, ratio ramps base -> max across federated rounds
    "ssl_mask_curriculum": False,
    "ssl_mask_ratio_max": 0.5,
    # 小料2: client-adaptive, scale ratio by local sample count vs per-client mean
    "ssl_mask_adaptive": False,
    "ssl_mask_adaptive_lo": 0.5,
    "ssl_mask_adaptive_hi": 1.5,
}


class GaussianBlurTransform:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if torch.rand(1).item() >= self.p:
            return img
        sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))


class SolarizeTransform:
    def __init__(self, p: float = 0.0):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if torch.rand(1).item() < self.p:
            return ImageOps.solarize(img)
        return img


class SSLImageDataset(Dataset):
    def __init__(self, records: List[Dict], image_size: int = 320):
        self.records = records
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlurTransform(p=0.5),
                SolarizeTransform(p=0.0),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record["image_path"]).convert("RGB")
        view_q = self.transform(image)
        view_k = self.transform(image)
        return {
            "view_q": view_q,
            "view_k": view_k,
            "image_path": record["image_path"],
            "source_dataset": record.get("source_dataset", "unknown"),
        }


class BalancedDatasetBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        records: Sequence[Dict],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
    ):
        self.batch_size = max(int(batch_size), 1)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.source_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, record in enumerate(records):
            self.source_to_indices[str(record.get("source_dataset", "unknown"))].append(idx)
        self.sources = sorted(self.source_to_indices)

    def __len__(self) -> int:
        num_items = sum(len(indices) for indices in self.source_to_indices.values())
        if self.drop_last:
            return num_items // self.batch_size
        return (num_items + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        self.epoch += 1

        pools: Dict[str, List[int]] = {}
        for source in self.sources:
            indices = list(self.source_to_indices[source])
            if self.shuffle:
                order = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[i] for i in order]
            pools[source] = indices

        cursors = {source: 0 for source in self.sources}
        source_cursor = 0
        while True:
            active = [source for source in self.sources if cursors[source] < len(pools[source])]
            if not active:
                break

            batch: List[int] = []
            attempts = 0
            while len(batch) < self.batch_size and active and attempts < self.batch_size * max(len(self.sources), 1) * 2:
                source = self.sources[source_cursor % len(self.sources)]
                source_cursor += 1
                attempts += 1
                if cursors[source] >= len(pools[source]):
                    active = [item for item in active if item != source]
                    continue
                batch.append(pools[source][cursors[source]])
                cursors[source] += 1

            if len(batch) == self.batch_size:
                yield batch
                continue
            if batch and not self.drop_last:
                yield batch
            break


class RTDETRBackboneEncoder(nn.Module):
    def __init__(self, model_variant: str = "RTDETR_L"):
        super().__init__()
        self.model_variant = model_variant
        self.detector = build_ssl_detector(model_variant, nc=20)
        self.backbone = self.detector.model[:10]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 2048
        self.last_feat: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        self.last_feat = features  # [B, C, h, w], cached for attentive masking
        pooled = self.pool(features)
        return torch.flatten(pooled, 1)


class MoCoRTDETR(nn.Module):
    def __init__(self, proj_dim: int, queue_size: int, temperature: float, model_variant: str = "RTDETR_L"):
        super().__init__()
        self.model_variant = model_variant
        self.encoder_q = RTDETRBackboneEncoder(model_variant=model_variant)
        self.encoder_k = RTDETRBackboneEncoder(model_variant=model_variant)
        self.projector_q = nn.Sequential(
            nn.Linear(self.encoder_q.out_dim, self.encoder_q.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_q.out_dim, proj_dim),
        )
        self.projector_k = nn.Sequential(
            nn.Linear(self.encoder_k.out_dim, self.encoder_k.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_k.out_dim, proj_dim),
        )
        self.temperature = temperature
        self.queue_size = queue_size

        self.register_buffer("queue", F.normalize(torch.randn(proj_dim, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self._sync_momentum_from_query()

    @torch.no_grad()
    def _sync_momentum_from_query(self) -> None:
        self.encoder_k.load_state_dict(self.encoder_q.state_dict(), strict=True)
        self.projector_k.load_state_dict(self.projector_q.state_dict(), strict=True)
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        for param in self.projector_k.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def momentum_update(self, momentum: float) -> None:
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1.0 - momentum)
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1.0 - momentum)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        keys = keys.detach()
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr.item())

        if batch_size >= self.queue_size:
            self.queue.copy_(keys[-self.queue_size :].T)
            self.queue_ptr[0] = 0
            return

        end = ptr + batch_size
        if end <= self.queue_size:
            self.queue[:, ptr:end] = keys.T
        else:
            first = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:first].T
            self.queue[:, : batch_size - first] = keys[first:].T
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def encode_query(self, images: torch.Tensor) -> torch.Tensor:
        q = self.projector_q(self.encoder_q(images))
        return F.normalize(q, dim=1)

    @torch.no_grad()
    def encode_key(self, images: torch.Tensor) -> torch.Tensor:
        k = self.projector_k(self.encoder_k(images))
        return F.normalize(k, dim=1)


def bool_config_value(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_ssl_config(run_config) -> Dict:
    config = dict(SSL_DEFAULTS)
    if run_config is not None:
        for key in SSL_DEFAULTS:
            if key in run_config:
                config[key] = run_config[key]
    config["ssl_enable_evaluate"] = bool_config_value(config["ssl_enable_evaluate"])
    config["ssl_balance_datasets_per_batch"] = bool_config_value(config["ssl_balance_datasets_per_batch"])
    config["ssl_mask_enable"] = bool_config_value(config["ssl_mask_enable"])
    config["ssl_mask_curriculum"] = bool_config_value(config["ssl_mask_curriculum"])
    config["ssl_mask_adaptive"] = bool_config_value(config["ssl_mask_adaptive"])
    return config


@torch.no_grad()
def _array_from_tensor(tensor: torch.Tensor) -> Array:
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return Array(arr)


def is_bn_key(module_name: str, model: nn.Module) -> bool:
    if not module_name:
        return False
    modules = dict(model.named_modules())
    module = modules.get(module_name)
    return isinstance(module, nn.modules.batchnorm._BatchNorm)


def get_ssl_federated_arrays(model: MoCoRTDETR) -> ArrayRecord:
    arrays: Dict[str, Array] = {}
    for name, param in model.encoder_q.named_parameters():
        arrays[f"encoder_q.{name}"] = _array_from_tensor(param)
    for name, param in model.projector_q.named_parameters():
        arrays[f"projector_q.{name}"] = _array_from_tensor(param)
    return ArrayRecord(arrays)


def load_ssl_federated_arrays(model: MoCoRTDETR, arrays: ArrayRecord) -> None:
    param_dict = {f"encoder_q.{name}": param for name, param in model.encoder_q.named_parameters()}
    param_dict.update({f"projector_q.{name}": param for name, param in model.projector_q.named_parameters()})

    for name, array in arrays.items():
        if name not in param_dict:
            continue
        target = param_dict[name]
        value = torch.from_numpy(array.numpy())
        if value.ndim == 1 and value.numel() == 1 and target.ndim == 0:
            value = value.reshape(())
        value = value.to(device=target.device, dtype=target.dtype)
        target.data.copy_(value)

    model._sync_momentum_from_query()


def get_ssl_local_state(model: MoCoRTDETR) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    named_modules = dict(model.encoder_q.named_modules())
    for name, tensor in model.encoder_q.state_dict().items():
        module_name = name.rsplit(".", 1)[0] if "." in name else ""
        if is_bn_key(module_name, model.encoder_q):
            state[f"encoder_q.{name}"] = tensor.detach().cpu().clone()
    return state


def load_ssl_local_state(model: MoCoRTDETR, local_state: Dict[str, torch.Tensor]) -> None:
    state_dict = model.encoder_q.state_dict()
    for name, tensor in local_state.items():
        key = name.replace("encoder_q.", "", 1)
        if key not in state_dict:
            continue
        target = state_dict[key]
        src = tensor.to(device=target.device, dtype=target.dtype)
        target.copy_(src)


class SSLLocalStateStore:
    def __init__(self, base_dir: Path = SSL_STATE_DIR):
        self.base_dir = Path(base_dir)

    def _state_path(self, run_id: str, mode_signature: str, client_id: int) -> Path:
        safe_mode = mode_signature.replace("|", "_").replace(":", "_")
        return self.base_dir / safe_mode / str(run_id) / f"client_{client_id}.pt"

    def save(self, run_id: str, mode_signature: str, client_id: int, state: Dict[str, torch.Tensor]) -> None:
        path = self._state_path(run_id, mode_signature, client_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    def load(self, run_id: str, mode_signature: str, client_id: int) -> Dict[str, torch.Tensor] | None:
        path = self._state_path(run_id, mode_signature, client_id)
        if not path.exists():
            return None
        return torch.load(path, map_location="cpu", weights_only=True)


_SSL_STATE_STORE = SSLLocalStateStore()


def get_ssl_mode_signature(config: Dict) -> str:
    return "|".join(
        [
            f"task:{config['ssl_mode']}",
            f"model:{config['ssl_model_variant']}",
            f"scope:{config['ssl_pretrain_scope']}",
            "strategy:FedAvg",
            f"dataset:{config['ssl_dataset']}",
            f"clients:{config['ssl_num_clients']}",
        ]
    )


def get_ssl_output_dir(config: Dict) -> Path:
    output_dir = Path("outputs") / (
        f"ssl_moco_{config['ssl_model_variant']}_FedAvg_"
        f"{config['ssl_dataset']}_{config['ssl_num_clients']}clients"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def parse_ssl_dataset_roots(config: Dict) -> List[Tuple[str, Path]]:
    raw = config.get("ssl_dataset_roots", "")
    if isinstance(raw, dict):
        items = [(str(name), Path(path)) for name, path in raw.items()]
    elif isinstance(raw, (list, tuple)):
        items = []
        for idx, item in enumerate(raw):
            if isinstance(item, dict):
                name = str(item.get("name", item.get("dataset", f"dataset_{idx}")))
                path = Path(str(item.get("path", item.get("root", ""))))
            else:
                text = str(item)
                name, path = _parse_dataset_root_spec(text, idx)
            items.append((name, path))
    else:
        text = str(raw).strip()
        if not text:
            return [(str(config.get("ssl_dataset", "dataset")), Path(str(config["ssl_dataset_root"])))]
        if text[0] in "[{":
            parsed = json.loads(text)
            return parse_ssl_dataset_roots({**config, "ssl_dataset_roots": parsed})
        parts = [part.strip() for part in text.replace("\n", ";").split(";") if part.strip()]
        items = [_parse_dataset_root_spec(part, idx) for idx, part in enumerate(parts)]

    missing = [(name, str(path)) for name, path in items if not path.exists()]
    if missing:
        raise FileNotFoundError(f"SSL dataset roots not found: {missing}")
    return items


def _parse_dataset_root_spec(text: str, idx: int) -> Tuple[str, Path]:
    if "=" in text:
        name, path = text.split("=", 1)
        return name.strip(), Path(path.strip())
    if "|" in text:
        name, path = text.split("|", 1)
        return name.strip(), Path(path.strip())
    path = Path(text.strip())
    return path.name or f"dataset_{idx}", path


def _make_ssl_image_record(image_path: Path, source_dataset: str) -> Dict:
    return {
        "image_path": str(image_path),
        "label_path": "",
        "labels": [],
        "class_ids": [],
        "primary_label": -1,
        "source_dataset": source_dataset,
    }


def _image_files_in_dir(image_dir: Path) -> List[Path]:
    if not image_dir.exists() or not image_dir.is_dir():
        return []
    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file()
        and not path.name.startswith("._")
        and "__MACOSX" not in path.parts
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    )


def discover_ssl_image_records(dataset_root: str | Path, split: str, source_dataset: str) -> List[Dict]:
    dataset_root = Path(dataset_root)

    voc_split_file = dataset_root / "ImageSets" / "Main" / f"{split}.txt"
    voc_image_dir = dataset_root / "JPEGImages"
    if voc_split_file.exists() and voc_image_dir.exists():
        records = []
        with voc_split_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                stem = line.strip().split()[0] if line.strip() else ""
                if not stem:
                    continue
                for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    image_path = voc_image_dir / f"{stem}{ext}"
                    if image_path.exists():
                        records.append(_make_ssl_image_record(image_path, source_dataset))
                        break
        if records:
            return records

    candidate_dirs = [
        dataset_root / "images" / split,
        dataset_root / split / "images",
        dataset_root / "images",
        dataset_root / split,
        dataset_root / "JPEGImages",
    ]
    for image_dir in candidate_dirs:
        image_paths = _image_files_in_dir(image_dir)
        if image_paths:
            return [_make_ssl_image_record(path, source_dataset) for path in image_paths]

    image_paths = sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        and not any("mask" in part.lower() for part in path.parts)
        and not any("__macosx" in part.lower() for part in path.parts)
    )
    return [_make_ssl_image_record(path, source_dataset) for path in image_paths]


def discover_ssl_records(config: Dict, split: str) -> Tuple[List[Dict], Dict[str, int]]:
    records: List[Dict] = []
    source_counts: Dict[str, int] = {}
    for source_dataset, root in parse_ssl_dataset_roots(config):
        source_records = discover_ssl_image_records(root, split, source_dataset)
        if not source_records and split not in {"train", "all"}:
            source_records = discover_ssl_image_records(root, "train", source_dataset)
        records.extend(source_records)
        source_counts[source_dataset] = len(source_records)
    if not records:
        roots = [(name, str(path)) for name, path in parse_ssl_dataset_roots(config)]
        raise FileNotFoundError(f"No SSL images found for split={split!r} under roots={roots}")
    return records, source_counts


def _source_from_mixed_ssl_name(image_path: Path) -> str:
    # Files generated by prepare_medical5_mixed_ssl80_label20.py use
    # source__hash__original_name.ext. Fall back to unknown for safety.
    return image_path.name.split("__", 1)[0] if "__" in image_path.name else "unknown"


def discover_fixed_client_ssl_records(config: Dict, partition_id: int, split: str) -> Tuple[List[Dict], Dict[str, int]]:
    root = Path(str(config["ssl_dataset_root"]))
    client_dir = root / f"client{int(partition_id)}" / "ssl_unlabeled" / "images" / split
    if not client_dir.exists() and split != "train":
        client_dir = root / f"client{int(partition_id)}" / "ssl_unlabeled" / "images" / "train"
    image_paths = _image_files_in_dir(client_dir)
    if not image_paths:
        raise FileNotFoundError(
            f"No fixed-client SSL images found for partition_id={partition_id}, "
            f"split={split!r}, dir={client_dir}"
        )
    records = [_make_ssl_image_record(path, _source_from_mixed_ssl_name(path)) for path in image_paths]
    source_counts: Dict[str, int] = {}
    for record in records:
        source = str(record.get("source_dataset", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1
    return records, dict(sorted(source_counts.items()))


def build_ssl_dataloader(
    partition_id: int,
    config: Dict,
    split: str = "train",
    shuffle: bool = True,
) -> Tuple[DataLoader, Dict]:
    loader_seed = (
        int(os.environ.get("MASTER_SEED", "42"))
        + int(partition_id) * 1000
        + int(config.get("_round_seed_offset", 0)) * 10000
        + (0 if split == "train" else 1_000_000)
    )
    generator = torch.Generator()
    generator.manual_seed(loader_seed)
    num_clients = max(int(config.get("ssl_num_clients", 1)), 1)
    partition_method = str(config.get("ssl_partition_method", "dirichlet")).lower()

    if partition_method in {"fixed_client_dirs", "fixed-client-dirs", "client_dirs", "client-dirs"}:
        records, source_counts = discover_fixed_client_ssl_records(config, int(partition_id), split)
        client_indices = list(range(len(records)))
        client_records = records
    else:
        if str(config.get("ssl_dataset_roots", "")).strip():
            records, source_counts = discover_ssl_records(config, split)
        else:
            records = discover_yolo_split_records(config["ssl_dataset_root"], split)
            for record in records:
                record.setdefault("source_dataset", str(config.get("ssl_dataset", "dataset")))
            source_counts = {str(config.get("ssl_dataset", "dataset")): len(records)}

        if num_clients <= 1 or partition_method in {"none", "global", "single"}:
            client_indices = list(range(len(records)))
        else:
            partitions = build_dirichlet_partitions(
                records,
                num_clients=num_clients,
                alpha=float(config["ssl_dirichlet_alpha"]),
                seed=loader_seed,
            )
            client_indices = partitions[int(partition_id)]

        client_records = [records[idx] for idx in client_indices]
    dataset = SSLImageDataset(client_records, image_size=int(config["ssl_image_size"]))
    balance_batches = bool_config_value(config.get("ssl_balance_datasets_per_batch", False))
    if balance_batches and shuffle and len(source_counts) > 1:
        batch_sampler = BalancedDatasetBatchSampler(
            client_records,
            batch_size=int(config["ssl_batch_size"]),
            shuffle=shuffle,
            drop_last=len(dataset) > 1,
            seed=loader_seed,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_dataloader_worker,
            generator=generator,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=int(config["ssl_batch_size"]),
            shuffle=shuffle,
            drop_last=len(dataset) > 1,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_dataloader_worker,
            generator=generator,
        )
    stats = {
        "split": split,
        "num_samples": len(dataset),
        "total_samples": len(records),
        "source_counts": source_counts,
        "client_source_counts": dict(
            sorted(
                {
                    source: sum(1 for record in client_records if record.get("source_dataset") == source)
                    for source in source_counts
                }.items()
            )
        ),
        "balanced_batches": bool(balance_batches and shuffle and len(source_counts) > 1),
        "partition_hist": summarize_partition(records, client_indices),
    }
    return loader, stats


def extract_state_dict_from_checkpoint(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "ema" in checkpoint and checkpoint["ema"] is not None:
        source_model = checkpoint["ema"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        source_model = checkpoint["model"]
    else:
        source_model = checkpoint

    if hasattr(source_model, "state_dict"):
        return source_model.state_dict()
    return source_model


def _strip_detector_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("detector."):
            new_key = new_key.replace("detector.", "", 1)
        normalized[new_key] = value
    return normalized


def load_pretrained_backbone(model: MoCoRTDETR, checkpoint_path: str) -> Dict[str, int]:
    source_state = extract_state_dict_from_checkpoint(checkpoint_path)
    target_state = model.encoder_q.state_dict()
    filtered: Dict[str, torch.Tensor] = {}

    for key, value in source_state.items():
        if not key.startswith("model."):
            continue
        parts = key.split(".")
        if len(parts) < 3:
            continue
        try:
            layer_idx = int(parts[1])
        except ValueError:
            continue
        if layer_idx > 9:
            continue
        target_key = f"detector.{key}"
        if target_key in target_state and target_state[target_key].shape == value.shape:
            filtered[target_key] = value

    model.encoder_q.load_state_dict(filtered, strict=False)
    model._sync_momentum_from_query()
    return {"loaded": len(filtered), "target": len(target_state)}


@torch.no_grad()
def ssl_evaluate(
    model: MoCoRTDETR,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_std = 0.0
    total_abs_cosine = 0.0

    for batch in dataloader:
        view_q = batch["view_q"].to(device, non_blocking=True)
        view_k = batch["view_k"].to(device, non_blocking=True)
        q = model.encode_query(view_q)
        k = model.encode_key(view_k)

        positive_logits = torch.sum(q * k, dim=1, keepdim=True)
        negative_logits = torch.matmul(q, model.queue.clone().detach())
        logits = torch.cat([positive_logits, negative_logits], dim=1) / model.temperature
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, targets)

        total_loss += float(loss.item())
        total_std += float(q.std(dim=0).mean().item())
        if q.size(0) > 1:
            cosine = torch.matmul(q, q.T).abs()
            off_diagonal = cosine[~torch.eye(q.size(0), dtype=torch.bool, device=q.device)]
            total_abs_cosine += float(off_diagonal.mean().item())
        total_batches += 1

        if max_batches > 0 and total_batches >= max_batches:
            break

    return {
        "val_ssl_loss": total_loss / max(total_batches, 1),
        "embedding_std": total_std / max(total_batches, 1),
        "embedding_abs_cosine": total_abs_cosine / max(total_batches, 1),
        "eval_batches": float(total_batches),
    }


@torch.no_grad()
def attentive_mask(
    model: MoCoRTDETR,
    view: torch.Tensor,
    ratio: float,
    mode: str = "foreground",
    randomness: float = 0.5,
) -> torch.Tensor:
    """Attentive masking on the *query* view (asymmetric: key view stays intact).

    The momentum key encoder gives a stable response map; cells with high response
    are treated as foreground. In "foreground" mode those cells are masked, forcing
    the query to match the full key from context. Returns a masked copy; no-op when
    ratio <= 0. Mask grid == backbone feature-map grid (aligned with stride).
    """
    if ratio <= 0.0:
        return view
    model.encoder_k(view)  # populate last_feat (no grad, momentum encoder)
    feat = model.encoder_k.last_feat
    if feat is None:
        return view
    b, _, h, w = feat.shape
    n_cells = h * w
    score = feat.float().pow(2).mean(dim=1).flatten(1)  # [B, h*w] response strength
    if mode == "background":
        score = -score
    score = torch.softmax(score, dim=1)
    # uniform noise so we don't always mask the exact same cells across steps
    rank = score + torch.rand_like(score) * (randomness / float(n_cells))
    n_mask = max(0, min(n_cells - 1, int(round(ratio * n_cells))))
    if n_mask == 0:
        return view
    idx = rank.topk(n_mask, dim=1).indices
    keep = torch.ones(b, n_cells, device=view.device, dtype=view.dtype)
    keep.scatter_(1, idx, 0.0)
    keep = keep.view(b, 1, h, w)
    keep = F.interpolate(keep, size=view.shape[-2:], mode="nearest")
    return view * keep


def resolve_mask_ratio(
    config: Dict,
    round_num: int = 1,
    total_rounds: int = 1,
    num_samples: int = 0,
    mean_samples: float = 0.0,
) -> float:
    """Base ratio + optional curriculum (小料1) + client-adaptive scaling (小料2).

    Safely degrades to the base ratio when round/sample info is unavailable.
    """
    ratio = float(config.get("ssl_mask_ratio", 0.3))
    if bool_config_value(config.get("ssl_mask_curriculum", False)) and total_rounds > 1:
        progress = max(0.0, min(1.0, (round_num - 1) / (total_rounds - 1)))
        ratio_max = float(config.get("ssl_mask_ratio_max", ratio))
        ratio = ratio + (ratio_max - ratio) * progress
    if bool_config_value(config.get("ssl_mask_adaptive", False)) and mean_samples > 0 and num_samples > 0:
        lo = float(config.get("ssl_mask_adaptive_lo", 0.5))
        hi = float(config.get("ssl_mask_adaptive_hi", 1.5))
        scale = max(lo, min(hi, num_samples / mean_samples))
        ratio = ratio * scale
    return max(0.0, min(0.9, ratio))


def build_mask_config(
    config: Dict,
    round_num: int = 1,
    total_rounds: int = 1,
    num_samples: int = 0,
    mean_samples: float = 0.0,
) -> Dict:
    """Resolve per-round, per-client mask settings for ssl_train_one_round.

    Returns {"enable": False} when the master switch is off, keeping training
    bit-identical to the plain-MoCo baseline.
    """
    if not bool_config_value(config.get("ssl_mask_enable", True)):
        return {"enable": False}
    return {
        "enable": True,
        "mode": str(config.get("ssl_mask_mode", "foreground")),
        "ratio": resolve_mask_ratio(config, round_num, total_rounds, num_samples, mean_samples),
        "randomness": float(config.get("ssl_mask_randomness", 0.5)),
    }


def ssl_train_one_round(
    model: MoCoRTDETR,
    trainloader: DataLoader,
    local_epochs: int,
    lr: float,
    momentum: float,
    device: torch.device,
    mask_config: Dict | None = None,
    max_batches: int = 0,
) -> Dict[str, float]:
    model.train()
    mask_enabled = bool(mask_config) and bool(mask_config.get("enable", False))
    mask_mode = str(mask_config.get("mode", "foreground")) if mask_enabled else "foreground"
    mask_ratio = float(mask_config.get("ratio", 0.0)) if mask_enabled else 0.0
    mask_randomness = float(mask_config.get("randomness", 0.5)) if mask_enabled else 0.0
    optimizer = torch.optim.AdamW(
        list(model.encoder_q.parameters()) + list(model.projector_q.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    total_loss = 0.0
    total_batches = 0
    total_norm = 0.0

    stop_training = False
    for _ in range(local_epochs):
        for batch in trainloader:
            view_q = batch["view_q"].to(device, non_blocking=True)
            view_k = batch["view_k"].to(device, non_blocking=True)

            if mask_enabled:
                view_q = attentive_mask(model, view_q, mask_ratio, mask_mode, mask_randomness)

            q = model.encode_query(view_q)
            with torch.no_grad():
                model.momentum_update(momentum)
                k = model.encode_key(view_k)

            positive_logits = torch.sum(q * k, dim=1, keepdim=True)
            negative_logits = torch.matmul(q, model.queue.clone().detach())
            logits = torch.cat([positive_logits, negative_logits], dim=1)
            logits = logits / model.temperature
            targets = torch.zeros(logits.size(0), dtype=torch.long, device=device)

            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.encoder_q.parameters()) + list(model.projector_q.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            model.dequeue_and_enqueue(k)

            total_loss += float(loss.item())
            total_batches += 1
            total_norm += float(q.norm(dim=1).mean().item())
            if max_batches > 0 and total_batches >= max_batches:
                stop_training = True
                break
        if stop_training:
            break

    avg_loss = total_loss / max(total_batches, 1)
    avg_norm = total_norm / max(total_batches, 1)
    queue_usage = 1.0 if total_batches > 0 else 0.0
    return {
        "ssl_loss": avg_loss,
        "embedding_norm": avg_norm,
        "queue_usage": queue_usage,
        "mask_enabled": 1.0 if mask_enabled else 0.0,
        "mask_ratio": mask_ratio,
        "train_batches": float(total_batches),
    }


def create_ssl_model(config: Dict, device: torch.device) -> MoCoRTDETR:
    model = MoCoRTDETR(
        proj_dim=int(config["ssl_proj_dim"]),
        queue_size=int(config["ssl_queue_size"]),
        temperature=float(config["ssl_temperature"]),
        model_variant=str(config["ssl_model_variant"]),
    )
    model.to(device)
    return model


def _cpu_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def _save_ssl_artifact_pair(
    model: MoCoRTDETR,
    config: Dict,
    output_dir: Path,
    backbone_name: str,
    full_name: str,
    meta_name: str | None = None,
    server_round: int | None = None,
) -> None:
    backbone_path = output_dir / backbone_name
    full_path = output_dir / full_name

    stripped_backbone = _strip_detector_prefix(_cpu_state_dict(model.encoder_q.state_dict()))
    torch.save(stripped_backbone, backbone_path)
    torch.save(
        {
            "encoder_q_backbone": stripped_backbone,
            "encoder_k_backbone": copy.deepcopy(stripped_backbone),
            "projector_q": _cpu_state_dict(model.projector_q.state_dict()),
            "projector_k": _cpu_state_dict(model.projector_q.state_dict()),
            "queue": model.queue.detach().cpu().clone(),
            "queue_ptr": model.queue_ptr.detach().cpu().clone(),
            "config": config,
            "server_round": server_round,
        },
        full_path,
    )

    if meta_name is not None:
        meta_path = output_dir / meta_name
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "mode_signature": get_ssl_mode_signature(config),
                    "output_dir": str(output_dir),
                    "backbone_ckpt": str(backbone_path),
                    "full_moco_ckpt": str(full_path),
                    "server_round": server_round,
                    "config": config,
                },
                handle,
                indent=2,
            )


def save_ssl_round_artifacts(model: MoCoRTDETR, config: Dict, output_dir: Path, server_round: int) -> None:
    _save_ssl_artifact_pair(
        model,
        config,
        output_dir,
        backbone_name=f"ssl_round_{server_round:03d}_backbone.pt",
        full_name=f"ssl_round_{server_round:03d}_full.pt",
        meta_name=f"ssl_round_{server_round:03d}_meta.json",
        server_round=server_round,
    )


def save_ssl_server_artifacts(model: MoCoRTDETR, config: Dict, output_dir: Path) -> None:
    resume_start_round = int(config.get("ssl_resume_start_round", 0) or 0)
    server_round = resume_start_round + int(config.get("num-server-rounds", 0))
    _save_ssl_artifact_pair(
        model,
        config,
        output_dir,
        backbone_name="ssl_global_backbone.pt",
        full_name="ssl_full_moco_last.pt",
        meta_name="ssl_meta.json",
        server_round=server_round,
    )


def load_ssl_full_checkpoint(model: MoCoRTDETR, checkpoint_path: str) -> Dict[str, int | str | None]:
    if not checkpoint_path:
        return {"loaded": 0, "checkpoint": None, "server_round": None}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported SSL full checkpoint format: {type(checkpoint)!r}")

    loaded = 0
    backbone = checkpoint.get("encoder_q_backbone")
    if isinstance(backbone, dict):
        target_state = model.encoder_q.state_dict()
        filtered: Dict[str, torch.Tensor] = {}
        for key, value in backbone.items():
            candidates = [key, f"detector.{key}"]
            for target_key in candidates:
                if target_key in target_state and target_state[target_key].shape == value.shape:
                    filtered[target_key] = value
                    break
        model.encoder_q.load_state_dict(filtered, strict=False)
        loaded += len(filtered)

    projector = checkpoint.get("projector_q")
    if isinstance(projector, dict):
        result = model.projector_q.load_state_dict(projector, strict=False)
        loaded += len(projector) - len(result.unexpected_keys)

    queue = checkpoint.get("queue")
    if torch.is_tensor(queue) and queue.shape == model.queue.shape:
        model.queue.copy_(queue.to(device=model.queue.device, dtype=model.queue.dtype))
    queue_ptr = checkpoint.get("queue_ptr")
    if torch.is_tensor(queue_ptr) and queue_ptr.shape == model.queue_ptr.shape:
        model.queue_ptr.copy_(queue_ptr.to(device=model.queue_ptr.device, dtype=model.queue_ptr.dtype))

    model._sync_momentum_from_query()
    return {"loaded": loaded, "checkpoint": checkpoint_path, "server_round": checkpoint.get("server_round")}


def restore_ssl_local_bn_state(model: MoCoRTDETR, run_id: str, mode_signature: str, client_id: int) -> bool:
    local_state = _SSL_STATE_STORE.load(run_id, mode_signature, client_id)
    if local_state is None:
        return False
    load_ssl_local_state(model, local_state)
    return True


def cache_ssl_local_bn_state(model: MoCoRTDETR, run_id: str, mode_signature: str, client_id: int) -> None:
    _SSL_STATE_STORE.save(run_id, mode_signature, client_id, get_ssl_local_state(model))
