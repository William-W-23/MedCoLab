from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VOC2007_CLASS_NAMES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def parse_yolo_label_file(label_path: Path) -> List[List[float]]:
    labels: List[List[float]] = []
    if not label_path.exists():
        return labels

    with label_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = raw_line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            labels.append([cls, x, y, w, h])
    return labels


def discover_yolo_split_records(dataset_root: str | Path, split: str) -> List[Dict]:
    dataset_root = Path(dataset_root)
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split

    records: List[Dict] = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        labels = parse_yolo_label_file(label_path)
        class_ids = [int(item[0]) for item in labels]
        records.append(
            {
                "image_path": str(image_path),
                "label_path": str(label_path),
                "labels": labels,
                "class_ids": class_ids,
                "primary_label": class_ids[0] if class_ids else -1,
            }
        )
    return records


def _counts_from_proportions(total: int, proportions: np.ndarray) -> np.ndarray:
    counts = np.floor(proportions * total).astype(int)
    remainder = total - int(counts.sum())
    if remainder <= 0:
        return counts

    ranking = np.argsort(-(proportions * total - counts))
    for idx in ranking[:remainder]:
        counts[idx] += 1
    return counts


def build_dirichlet_partitions(
    records: Sequence[Dict],
    num_clients: int,
    alpha: float,
    seed: int,
) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        label_to_indices[int(record.get("primary_label", -1))].append(idx)

    partitions: List[List[int]] = [[] for _ in range(num_clients)]
    for label, indices in sorted(label_to_indices.items(), key=lambda item: item[0]):
        indices = list(indices)
        rng.shuffle(indices)

        if label == -1:
            split_indices = np.array_split(np.asarray(indices, dtype=int), num_clients)
            for client_id, client_indices in enumerate(split_indices):
                partitions[client_id].extend(client_indices.tolist())
            continue

        proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
        counts = _counts_from_proportions(len(indices), proportions)
        start = 0
        for client_id, count in enumerate(counts.tolist()):
            end = start + count
            partitions[client_id].extend(indices[start:end])
            start = end

    non_empty = [idx for idx, part in enumerate(partitions) if part]
    if non_empty:
        largest_client = max(range(num_clients), key=lambda idx: len(partitions[idx]))
        for client_id, part in enumerate(partitions):
            if part:
                rng.shuffle(part)
                continue
            moved_index = partitions[largest_client].pop()
            part.append(moved_index)
            rng.shuffle(partitions[largest_client])

    for part in partitions:
        rng.shuffle(part)
    return partitions


def build_iid_partitions(num_items: int, num_clients: int, seed: int) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(num_items, dtype=int)
    rng.shuffle(indices)
    return [chunk.tolist() for chunk in np.array_split(indices, num_clients)]


def sample_client_indices(indices: Sequence[int], fraction: float, seed: int) -> List[int]:
    indices = list(indices)
    if not indices:
        return []
    if fraction >= 1.0:
        return indices
    if fraction <= 0:
        raise ValueError("fraction must be > 0")

    rng = np.random.default_rng(seed)
    sample_size = max(1, int(round(len(indices) * fraction)))
    chosen = rng.choice(indices, size=min(sample_size, len(indices)), replace=False)
    return sorted(int(item) for item in chosen.tolist())


def summarize_partition(records: Sequence[Dict], indices: Sequence[int]) -> Dict[int, int]:
    label_hist: Dict[int, int] = defaultdict(int)
    for idx in indices:
        class_ids = records[idx].get("class_ids", [])
        if not class_ids:
            label_hist[-1] += 1
            continue
        for class_id in set(class_ids):
            label_hist[int(class_id)] += 1
    return dict(sorted(label_hist.items(), key=lambda item: item[0]))
