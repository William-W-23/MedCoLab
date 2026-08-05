#!/usr/bin/env python3
"""Create a sample-count-weighted Round 0 backbone from five independent SSL models."""

import argparse
import json
from pathlib import Path

import torch

from fl.classification_task import MultiDatasetRTDETRClassifier, load_round0_backbone, sha256_file

DEFAULT_SSL_COUNTS = [75439, 75361, 75440, 75430, 75449]
DEFAULT_MANIFEST = "de2451a9c5ca16e835901412c105a636dc80ede384dded39f12e7867c86b2233"  # pragma: allowlist secret


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-prefix", default="medical5_classification")
    parser.add_argument("--manifest-sha256", default=DEFAULT_MANIFEST)
    parser.add_argument("--ssl-counts", default=",".join(map(str, DEFAULT_SSL_COUNTS)))
    args = parser.parse_args()
    ssl_counts = [int(value) for value in args.ssl_counts.split(",")]
    if len(ssl_counts) != 5 or any(value <= 0 for value in ssl_counts):
        raise ValueError("--ssl-counts must contain five positive comma-separated counts")
    paths = [args.outputs_root / f"ssl_moco_RTDETR_L_Independent_{args.task_prefix}_client{i}_seed42" / "ssl_best_by_loss_backbone.pt" for i in range(5)]
    states, metadata = [], []
    for i, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        meta = json.loads((path.parent / "ssl_best_by_loss_meta.json").read_text(encoding="utf-8"))
        config = dict(meta.get("config", {}))
        if int(config.get("master_seed", -1)) != 42 or config.get("data_manifest_sha256") != args.manifest_sha256:
            raise RuntimeError(f"client{i} seed/manifest mismatch")
        raw_state = torch.load(path, map_location="cpu", weights_only=False)
        backbone_state = {}
        for key, value in raw_state.items():
            if not key.startswith("model."):
                continue
            parts = key.split(".")
            if len(parts) >= 3 and parts[1].isdigit() and int(parts[1]) <= 9:
                backbone_state[key] = value
        if len(backbone_state) != 480:
            raise RuntimeError(f"client{i} expected 480 true backbone tensors, got {len(backbone_state)}")
        states.append(backbone_state)
        metadata.append({
            "client": i, "samples": ssl_counts[i], "checkpoint": str(path),
            "checkpoint_sha256": sha256_file(path), "best_round": int(meta["server_round"]),
            "best_ssl_loss": float(meta["best_ssl_loss"]),
            "corrected_mode_signature": config.get("mode_signature"),
            "artifact_tensor_count": len(raw_state),
            "used_backbone_tensor_count": len(backbone_state),
        })
    keys = list(states[0])
    for i, state in enumerate(states[1:], start=1):
        if list(state) != keys:
            raise RuntimeError(f"client{i} key mismatch")
        for key in keys:
            if tuple(state[key].shape) != tuple(states[0][key].shape):
                raise RuntimeError(f"client{i} shape mismatch at {key}")
    total = float(sum(ssl_counts))
    averaged = {}
    for key in keys:
        first = states[0][key]
        if torch.is_floating_point(first):
            accumulator = torch.zeros_like(first, dtype=torch.float32)
            for state, count in zip(states, ssl_counts):
                accumulator.add_(state[key].float(), alpha=float(count) / total)
            averaged[key] = accumulator.to(first.dtype)
        else:
            averaged[key] = torch.stack([state[key] for state in states]).max(dim=0).values
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "weighted_ssl_round0_backbone.pt"
    torch.save(averaged, output_path)
    load_report = load_round0_backbone(MultiDatasetRTDETRClassifier(dropout=0.1), output_path)
    if int(load_report["loaded"]) != 480:
        raise RuntimeError(f"Round 0 load failed: {load_report}")
    report = {
        "strategy": "ssl_sample_count_weighted_backbone_average",
        "manifest_sha256": args.manifest_sha256, "master_seed": 42, "clients": metadata,
        "total_ssl_samples": int(total), "output": str(output_path),
        "output_sha256": sha256_file(output_path), "tensor_count": len(averaged),
        "load_report": load_report,
        "metadata_note": "Independent is authoritative; legacy outer FedAvg clients=1 signature is not propagated.",
    }
    (args.output_dir / "weighted_ssl_round0_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
