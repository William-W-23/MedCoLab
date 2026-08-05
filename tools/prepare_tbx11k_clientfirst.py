#!/usr/bin/env python3
"""Prepare public TBX11K train+validation data for the five-client protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


CLASS_NAMES = {0: "active_tb", 1: "latent_tb"}
COCO_CLASS_ID = {1: 0, 2: 1}
EXPECTED_IMAGES = 8976
EXPECTED_ANNOTATIONS = 1211
LABELED_FRACTION = 0.20
POSITIVE_LABELED_FRACTION = 0.80


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_name(relative: str) -> str:
    if relative.startswith(("tb/", "health/", "sick/")):
        return "tbx11k_main"
    basename = Path(relative).name
    if relative.startswith("extra/da+db/"):
        return "db" if basename.lower().endswith(".jpg") else "da"
    if relative.startswith("extra/mc+shenzhen/"):
        if basename.startswith("MCUCXR_"):
            return "montgomery"
        if basename.startswith("CHNCXR_"):
            return "shenzhen"
    raise ValueError(f"Cannot identify source for {relative}")


def safe_name(relative: str) -> str:
    path = Path(relative)
    token = hashlib.sha1(relative.encode()).hexdigest()[:10]
    prefix = "_".join(path.parts[:-1]).replace("+", "_")
    return f"{prefix}__{path.stem}__{token}{path.suffix.lower()}"


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source)


def balanced_random_clients(records: list[dict], rng: random.Random) -> dict[str, int]:
    targets = [len(records) // 5 + (1 if i < len(records) % 5 else 0) for i in range(5)]
    expected_sources = {"tbx11k_main", "da", "db", "montgomery", "shenzhen"}
    for _ in range(10000):
        order = list(records)
        rng.shuffle(order)
        slots = [client for client, count in enumerate(targets) for _ in range(count)]
        rng.shuffle(slots)
        owner = {record["relative"]: client for record, client in zip(order, slots)}
        sources = {client: set() for client in range(5)}
        classes = {client: Counter() for client in range(5)}
        for record in records:
            client = owner[record["relative"]]
            sources[client].add(record["source"])
            classes[client].update(record["classes"])
        if not all(sources[c] == expected_sources for c in range(5)):
            continue
        hist = [tuple(classes[c][k] for k in (0, 1)) for c in range(5)]
        if all(classes[c][0] and classes[c][1] for c in range(5)) and len(set(hist)) > 1:
            return owner
    raise RuntimeError("Unable to allocate five near-equal non-IID clients with all five sources")


def choose_local_splits(records: list[dict], rng: random.Random) -> dict[str, str]:
    """Keep 80:20 exactly while enriching the labeled pool with 80% of positives."""
    labeled_target = round(len(records) * LABELED_FRACTION)
    positive_groups: dict[tuple[int, ...], list[dict]] = defaultdict(list)
    eligible_negatives = []
    for record in records:
        if record["source"] != "tbx11k_main":
            continue
        if record["classes"]:
            positive_groups[tuple(record["classes"])].append(record)
        elif record["relative"].startswith(("health/", "sick/")):
            eligible_negatives.append(record)

    selected = []
    for signature, group in sorted(positive_groups.items()):
        rng.shuffle(group)
        count = max(3, round(len(group) * POSITIVE_LABELED_FRACTION))
        selected.extend(group[: min(count, len(group))])
    if len(selected) > labeled_target:
        raise RuntimeError(f"Positive selection exceeds labeled budget: {len(selected)}>{labeled_target}")
    rng.shuffle(eligible_negatives)
    selected.extend(eligible_negatives[: labeled_target - len(selected)])
    if len(selected) != labeled_target:
        raise RuntimeError(f"Insufficient fully annotated images for labeled pool: {len(selected)}")

    target_counts = [
        round(labeled_target * 0.7),
        round(labeled_target * 0.1),
    ]
    target_counts.append(labeled_target - sum(target_counts))
    names = ("train", "val", "test")
    for _ in range(10000):
        order = list(selected)
        rng.shuffle(order)
        split_of = {}
        offset = 0
        for split, count in zip(names, target_counts):
            for record in order[offset : offset + count]:
                split_of[record["relative"]] = split
            offset += count
        coverage = {split: set() for split in names}
        for record in selected:
            coverage[split_of[record["relative"]]].update(record["classes"])
        class_images = {
            split: Counter(class_id for record in selected if split_of[record["relative"]] == split for class_id in record["classes"])
            for split in names
        }
        if all(coverage[split] == {0, 1} for split in names) and all(
            class_images[split][class_id] >= 2 for split in ("val", "test") for class_id in (0, 1)
        ):
            return {record["relative"]: split_of.get(record["relative"], "ssl") for record in records}
    raise RuntimeError("Unable to create 7:1:2 labeled splits with both TB classes in every split")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source = args.source.resolve()
    annotation_path = source / "annotations" / "json" / "all_trainval.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)

    staging = args.output.parent / f".{args.output.name}.staging"
    if args.output.exists():
        marker = args.output / "preparation_complete.json"
        if marker.is_file():
            print(marker.read_text())
            return
        raise RuntimeError(f"Refusing to overwrite incomplete output: {args.output}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    coco = json.loads(annotation_path.read_text())
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    if len(images) != EXPECTED_IMAGES or len(annotations) != EXPECTED_ANNOTATIONS:
        raise RuntimeError(
            f"Unexpected official public inventory: images={len(images)} annotations={len(annotations)}"
        )
    category_ids = {int(item["id"]) for item in coco.get("categories", [])}
    if not {1, 2}.issubset(category_ids):
        raise RuntimeError(f"Missing active/latent categories: {category_ids}")

    image_by_id = {int(item["id"]): item for item in images}
    boxes_by_image: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if category_id not in COCO_CLASS_ID:
            raise RuntimeError(f"Unsupported annotated category {category_id}")
        x, y, width, height = map(float, annotation["bbox"])
        image_meta = image_by_id[image_id]
        image_width, image_height = int(image_meta["width"]), int(image_meta["height"])
        if not (width > 0 and height > 0 and x >= 0 and y >= 0 and x + width <= image_width + 1e-4 and y + height <= image_height + 1e-4):
            raise RuntimeError(f"Invalid COCO box image={image_id}: {annotation['bbox']}")
        boxes_by_image[image_id].append((COCO_CLASS_ID[category_id], x, y, width, height))

    records = []
    modes = Counter()
    for item in images:
        relative = str(item["file_name"])
        path = (source / "imgs" / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            width, height = image.size
            modes[image.mode] += 1
        if (width, height) != (int(item["width"]), int(item["height"])):
            raise RuntimeError(f"Image/annotation size mismatch: {relative}")
        boxes = boxes_by_image.get(int(item["id"]), [])
        source_id = source_name(relative)
        if source_id != "tbx11k_main" and boxes:
            raise RuntimeError(f"External classification-only image unexpectedly has a box: {relative}")
        classes = sorted({box[0] for box in boxes})
        records.append(
            {
                "relative": relative,
                "path": path,
                "source": source_id,
                "width": width,
                "height": height,
                "boxes": boxes,
                "classes": classes,
            }
        )
    if len({record["relative"] for record in records}) != len(records):
        raise RuntimeError("Duplicate image identifiers in all_trainval.json")
    source_inventory = Counter(record["source"] for record in records)
    expected_sources = {"tbx11k_main", "da", "db", "montgomery", "shenzhen"}
    if set(source_inventory) != expected_sources:
        raise RuntimeError(f"Unexpected source inventory: {source_inventory}")

    rng = random.Random(args.seed)
    client_of = balanced_random_clients(records, rng)
    split_of = {}
    for client in range(5):
        local = [record for record in records if client_of[record["relative"]] == client]
        split_of.update(choose_local_splits(local, rng))

    manifest_rows = []
    class_counts = {client: {split: Counter() for split in ("ssl", "train", "val", "test")} for client in range(5)}
    source_counts = {client: Counter() for client in range(5)}
    split_counts = {client: Counter() for client in range(5)}
    for record in sorted(records, key=lambda row: row["relative"]):
        relative = record["relative"]
        client = client_of[relative]
        split = split_of[relative]
        filename = safe_name(relative)
        if split == "ssl":
            destination = staging / "independent_ssl_views" / f"model_client{client}" / "client0" / "ssl_unlabeled" / "images" / "train" / filename
        else:
            destination = staging / f"client{client}" / "labeled" / "images" / split / filename
        link(record["path"], destination)
        if split != "ssl":
            label_path = staging / f"client{client}" / "labeled" / "labels" / split / f"{Path(filename).stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for class_id, x, y, width, height in sorted(record["boxes"]):
                xc = (x + width / 2) / record["width"]
                yc = (y + height / 2) / record["height"]
                bw = width / record["width"]
                bh = height / record["height"]
                if not all(0 < value <= 1 for value in (xc, yc, bw, bh)):
                    raise RuntimeError(f"Out-of-range YOLO box: {relative}")
                lines.append(f"{class_id} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        class_counts[client][split].update(record["classes"])
        source_counts[client][record["source"]] += 1
        split_counts[client][split] += 1
        manifest_rows.append(
            {
                "image": str(args.output / destination.relative_to(staging)),
                "source": record["source"],
                "patient_group": f"{record['source']}::{relative}",
                "client": client,
                "split": split,
                "classes": ";".join(map(str, record["classes"])),
                "boxes": len(record["boxes"]),
                "official_pool": "all_trainval",
                "source_image": str(record["path"]),
            }
        )

    for client in range(5):
        if set(source_counts[client]) != expected_sources:
            raise RuntimeError(f"Missing source coverage client={client}: {source_counts[client]}")
        total = sum(split_counts[client].values())
        labeled = total - split_counts[client]["ssl"]
        if labeled != round(total * LABELED_FRACTION):
            raise RuntimeError(f"Client {client} violates 80:20 split: {split_counts[client]}")
        for split in ("ssl", "train", "val", "test"):
            if set(class_counts[client][split]) != {0, 1}:
                raise RuntimeError(f"Missing class coverage client={client} split={split}: {class_counts[client][split]}")
    owners = defaultdict(set)
    for row in manifest_rows:
        owners[row["patient_group"]].add((row["client"], row["split"]))
    leaks = {key: value for key, value in owners.items() if len(value) != 1}
    if leaks:
        raise RuntimeError(f"Group leakage detected: {list(leaks.items())[:3]}")
    labeled_hist = [
        tuple(sum(class_counts[c][s][k] for s in ("train", "val", "test")) for k in (0, 1))
        for c in range(5)
    ]
    if len(set(labeled_hist)) == 1:
        raise RuntimeError("Client labeled class distributions are exactly balanced")

    manifest = staging / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    manifest_digest = sha256(manifest)
    (staging / "manifest.sha256").write_text(manifest_digest + "\n")
    report = {
        "dataset": "TBX11K public all_trainval",
        "paper": "Rethinking Computer-Aided Tuberculosis Diagnosis (CVPR 2020; revisited TPAMI 2023)",
        "protocol": "five-client source-covered non-IID allocation; local 80% SSL, 20% labeled; labeled 7:1:2",
        "seed": args.seed,
        "classes": {str(key): value for key, value in CLASS_NAMES.items()},
        "images": len(records),
        "groups": len(owners),
        "group_leaks": 0,
        "client_image_counts": {str(c): sum(split_counts[c].values()) for c in range(5)},
        "split_counts": {str(c): dict(split_counts[c]) for c in range(5)},
        "source_counts": {str(c): dict(source_counts[c]) for c in range(5)},
        "class_counts": {str(c): {s: dict(class_counts[c][s]) for s in class_counts[c]} for c in range(5)},
        "image_modes": dict(modes),
        "manifest_sha256": manifest_digest,
        "all_trainval_json_sha256": sha256(annotation_path),
        "source_policy": (
            "Only official public all_trainval is used. Official hidden-test images are excluded. "
            "The 576 DA/DB/Montgomery/Shenzhen images have no released detection boxes in this package "
            "and are therefore restricted to SSL; no positive image is silently treated as detection background."
        ),
        "group_policy": (
            "TBX11K and the four added sources contain one radiograph per released subject identifier; "
            "the manifest uses the immutable source-relative image identity as the isolation group."
        ),
        "labeled_positive_policy": (
            "The 20% labeled budget is preserved per client. Eighty percent of each positive class-signature "
            "is deterministically selected for labeled data, then the budget is filled with annotated main-source negatives."
        ),
        "channel_policy": "Official 512x512 files are retained byte-for-byte; loaders deterministically convert PIL images to RGB.",
        "site_policy": "Five source datasets are represented in every simulated client; no unavailable hospital ID is invented.",
    }
    (staging / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    (staging / "preparation_complete.json").write_text(json.dumps(report, indent=2) + "\n")
    staging.rename(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
