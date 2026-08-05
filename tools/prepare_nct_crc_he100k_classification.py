#!/usr/bin/env python3
"""Prepare a fixed five-client patch split with the official external test locked."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


CLASSES = ("ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")
CLIENTS = 5
MAIN_IMAGES = 100_000
EXTERNAL_IMAGES = 7_180
ARCHIVES = {
    "NCT-CRC-HE-100K.zip": (11_690_284_003, "6fd702d11df6292bc054397ae038a464"),  # pragma: allowlist secret
    "CRC-VAL-HE-7K.zip": (800_276_929, "2fd1651b4f94ebd818ebf90ad2b6ce06"),  # pragma: allowlist secret
}


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_archive(path: Path) -> None:
    expected_bytes, expected_md5 = ARCHIVES[path.name]
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(f"Archive byte-size mismatch: {path}")
    if digest(path, "md5") != expected_md5:
        raise RuntimeError(f"Archive MD5 mismatch: {path}")


def extract_one(archive: Path, destination: Path) -> None:
    marker = destination / "EXTRACTION_COMPLETE"
    if marker.is_file():
        return
    if destination.exists():
        raise FileExistsError(f"Unverified extraction directory exists: {destination}")
    staging = destination.with_name(destination.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as source:
            root = staging.resolve()
            for member in source.infolist():
                target = (staging / member.filename).resolve()
                if target != root and root not in target.parents:
                    raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
            source.extractall(staging)
        (staging / "EXTRACTION_COMPLETE").write_text(
            json.dumps({"archive": archive.name, "bytes": archive.stat().st_size, "md5": digest(archive, "md5")},
                       sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def inventory(root: Path, expected: int, collection: str, verify_images: bool) -> list[dict]:
    class_dirs = {path.name: path for path in root.iterdir() if path.is_dir()}
    if set(class_dirs) != set(CLASSES):
        raise RuntimeError(f"Unexpected class directories under {root}: {sorted(class_dirs)}")
    rows = []
    for label in CLASSES:
        paths = sorted(path for path in class_dirs[label].iterdir() if path.suffix.lower() in {".tif", ".tiff"})
        if not paths:
            raise RuntimeError(f"Empty class directory: {class_dirs[label]}")
        for path in paths:
            if not path.name.startswith(label + "-"):
                raise RuntimeError(f"Filename/class mismatch: {path}")
            if verify_images:
                with Image.open(path) as image:
                    if image.size != (224, 224):
                        raise RuntimeError(f"Unexpected image size {image.size}: {path}")
                    image.verify()
            rows.append({
                "dataset": "nct_crc_he100k",
                "collection": collection,
                "image_id": path.stem,
                "class": label,
                "source_path": str(path.resolve()),
                "sha256": digest(path),
            })
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} images under {root}, got {len(rows)}")
    if len({row["source_path"] for row in rows}) != len(rows):
        raise RuntimeError(f"Duplicate source paths under {root}")
    return rows


def assign(main_rows: list[dict], external_rows: list[dict], seed: int) -> None:
    for label in CLASSES:
        class_rows = [row for row in main_rows if row["class"] == label]
        random.Random(f"{seed}:main:{label}").shuffle(class_rows)
        by_client = [[] for _ in range(CLIENTS)]
        for index, row in enumerate(class_rows):
            by_client[index % CLIENTS].append(row)
        for client, client_rows in enumerate(by_client):
            random.Random(f"{seed}:split:{client}:{label}").shuffle(client_rows)
            ssl_n = round(len(client_rows) * 0.8)
            labeled = client_rows[ssl_n:]
            val_n = max(1, round(len(labeled) / 8))
            for row in client_rows[:ssl_n]:
                row.update(client=client, split="ssl")
            for row in labeled[:-val_n]:
                row.update(client=client, split="train")
            for row in labeled[-val_n:]:
                row.update(client=client, split="val")

        test_rows = [row for row in external_rows if row["class"] == label]
        random.Random(f"{seed}:external:{label}").shuffle(test_rows)
        for index, row in enumerate(test_rows):
            row.update(client=index % CLIENTS, split="test")


def safe_symlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source, target)


def write_outputs(rows: list[dict], output: Path, seed: int, archives: list[Path]) -> dict:
    complete = output / "PREPARATION_COMPLETE"
    if complete.is_file():
        return json.loads(complete.read_text(encoding="utf-8"))
    if output.exists():
        raise FileExistsError(f"Unverified prepared output exists: {output}")
    staging = output.with_name(output.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    metadata = staging / "metadata"
    metadata.mkdir(parents=True)
    manifest_path = metadata / "split_manifest.csv"
    fields = ("dataset", "collection", "image_id", "class", "client", "split", "source_path", "sha256")
    relative_paths = {}
    try:
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in sorted(rows, key=lambda value: (value["client"], value["split"], value["class"], value["image_id"])):
                writer.writerow({field: row[field] for field in fields})
                source = Path(row["source_path"])
                client_root = staging / f"client{row['client']}"
                if row["split"] == "ssl":
                    target = client_root / "ssl_unlabeled/images/train" / source.name
                else:
                    target = client_root / "labeled/images" / row["split"] / f"nct_crc_he100k__{row['class']}" / source.name
                safe_symlink(source, target)
                relative_paths[(row["collection"], row["image_id"])] = str(target.relative_to(staging))
        for client in range(CLIENTS):
            view = staging / f"independent_ssl_views/model_client{client}/client0/ssl_unlabeled/images/train"
            source = staging / f"client{client}/ssl_unlabeled/images/train"
            view.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(os.path.relpath(source, view.parent), view)

        manifest_sha = digest(manifest_path)
        counts = defaultdict(Counter)
        class_counts = defaultdict(Counter)
        hash_splits = defaultdict(set)
        hash_collections = defaultdict(set)
        hash_clients = defaultdict(set)
        collection_splits = defaultdict(Counter)
        for row in rows:
            client = f"client{row['client']}"
            key = f"{client}/{row['split']}"
            counts[client][row["split"]] += 1
            class_counts[key][row["class"]] += 1
            hash_splits[row["sha256"]].add(row["split"])
            hash_collections[row["sha256"]].add(row["collection"])
            hash_clients[row["sha256"]].add(client)
            collection_splits[row["collection"]][row["split"]] += 1
        broken = sum(1 for path in staging.rglob("*") if path.is_symlink() and not path.exists())
        cross_split_hashes = sum(len(values) > 1 for values in hash_splits.values())
        cross_collection_hashes = sum(len(values) > 1 for values in hash_collections.values())
        cross_client_hashes = sum(len(values) > 1 for values in hash_clients.values())
        source_boundary_violations = (
            collection_splits["NCT-CRC-HE-100K"]["test"]
            + sum(count for split, count in collection_splits["CRC-VAL-HE-7K"].items() if split != "test")
        )
        required_cells = [f"client{client}/{split}" for client in range(CLIENTS) for split in ("train", "val", "test")]
        missing_class_cells = [cell for cell in required_cells if set(class_counts[cell]) != set(CLASSES)]
        summary = {
            "dataset": "nct_crc_he100k",
            "seed": seed,
            "total_images": len(rows),
            "main_images": sum(row["collection"] == "NCT-CRC-HE-100K" for row in rows),
            "external_test_images": sum(row["collection"] == "CRC-VAL-HE-7K" for row in rows),
            "classes": list(CLASSES),
            "by_collection_class": {
                name: dict(sorted(Counter(row["class"] for row in rows if row["collection"] == name).items()))
                for name in ("NCT-CRC-HE-100K", "CRC-VAL-HE-7K")
            },
            "by_client_split_images": {key: dict(sorted(value.items())) for key, value in sorted(counts.items())},
            "by_client_split_class_images": {key: dict(sorted(value.items())) for key, value in sorted(class_counts.items())},
        }
        audit = {
            "archive_provenance": {
                path.name: {"bytes": path.stat().st_size, "md5": digest(path, "md5")} for path in archives
            },
            "source_manifest_sha256": manifest_sha,
            "duplicate_source_paths": len(rows) - len({row["source_path"] for row in rows}),
            "cross_split_duplicate_sha256": cross_split_hashes,
            "cross_collection_duplicate_sha256": cross_collection_hashes,
            "cross_client_duplicate_sha256": cross_client_hashes,
            "broken_symlinks": broken,
            "source_boundary_violations": source_boundary_violations,
            "missing_class_coverage_cells": missing_class_cells,
            "patient_id_available": False,
            "source_slide_id_available": False,
            "patient_slide_leakage_audit": "unavailable: official public patch archives expose no patch-to-patient or patch-to-slide mapping",
            "client_definition": "deterministic class-stratified synthetic patch partition; not an institution partition",
            "test_policy": "CRC-VAL-HE-7K only; excluded from SSL, train, validation, checkpoint selection, and retries",
        }
        if any((audit["duplicate_source_paths"], cross_split_hashes, cross_collection_hashes, cross_client_hashes,
                broken, source_boundary_violations, missing_class_cells)):
            raise RuntimeError(f"Integrity audit failed: {audit}")
        (metadata / "split_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (metadata / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        (metadata / "split_manifest.sha256").write_text(f"{manifest_sha}  split_manifest.csv\n")
        clients = {str(client): {f"{split}_paths": [] for split in ("train", "val", "test")} for client in range(CLIENTS)}
        for row in rows:
            if row["split"] != "ssl":
                clients[str(row["client"])][f"{row['split']}_paths"].append(relative_paths[(row["collection"], row["image_id"])])
        for client in clients.values():
            for key in client:
                client[key].sort()
        federated = {
            "schema": "nct_crc_he100k_external_test_patch_split_v1",
            "seed": seed,
            "ratios": {"main_ssl": 0.8, "main_labeled": 0.2, "labeled_train": 0.875, "labeled_val": 0.125},
            "external_test": "CRC-VAL-HE-7K",
            "source_manifest": str(output / "metadata/split_manifest.csv"),
            "source_manifest_sha256": manifest_sha,
            "patient_slide_grouping_available": False,
            "clients": clients,
        }
        federated_path = metadata / "federated_manifest.json"
        federated_path.write_text(json.dumps(federated, indent=2, sort_keys=True) + "\n")
        result = {
            "output": str(output),
            "source_manifest_sha256": manifest_sha,
            "federated_manifest_sha256": digest(federated_path),
            "ssl_counts": [counts[f"client{client}"]["ssl"] for client in range(CLIENTS)],
            "summary": summary,
            "audit": audit,
        }
        (staging / "PREPARATION_COMPLETE").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        staging.rename(output)
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-images", action="store_true")
    args = parser.parse_args()
    main_archive = args.downloads / "NCT-CRC-HE-100K.zip"
    external_archive = args.downloads / "CRC-VAL-HE-7K.zip"
    for archive in (main_archive, external_archive):
        verify_archive(archive)
    extract_one(main_archive, args.raw_root / "main")
    extract_one(external_archive, args.raw_root / "external")
    main_rows = inventory(args.raw_root / "main/NCT-CRC-HE-100K", MAIN_IMAGES, "NCT-CRC-HE-100K", args.verify_images)
    external_rows = inventory(args.raw_root / "external/CRC-VAL-HE-7K", EXTERNAL_IMAGES, "CRC-VAL-HE-7K", args.verify_images)
    assign(main_rows, external_rows, args.seed)
    result = write_outputs(main_rows + external_rows, args.output, args.seed, [main_archive, external_archive])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
