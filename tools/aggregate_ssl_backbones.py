#!/usr/bin/env python3
"""Sample-count-weighted aggregation of five independent SSL backbones."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs=5, required=True)
    parser.add_argument("--counts", nargs=5, type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if any(count <= 0 for count in args.counts):
        raise ValueError(f"All sample counts must be positive: {args.counts}")
    states = [torch.load(path, map_location="cpu", weights_only=False) for path in args.inputs]
    keys = list(states[0])
    if any(list(state) != keys for state in states[1:]):
        raise RuntimeError("SSL backbone key mismatch")

    total = sum(args.counts)
    largest = max(range(5), key=lambda index: (args.counts[index], -index))
    output_state = {}
    for key in keys:
        values = [state[key] for state in states]
        if not all(
            isinstance(value, torch.Tensor) and value.shape == values[0].shape
            for value in values
        ):
            raise RuntimeError(f"Tensor mismatch: {key}")
        if values[0].is_floating_point():
            output_state[key] = sum(
                value.to(torch.float64) * (count / total)
                for value, count in zip(values, args.counts)
            ).to(values[0].dtype)
        else:
            output_state[key] = values[largest].clone()

    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output / "weighted_ssl_backbone.pt"
    torch.save(output_state, target)
    metadata = {
        "method": "sample_count_weighted_average_of_best_ssl_backbones",
        "sample_counts": args.counts,
        "weights": [count / total for count in args.counts],
        "inputs": args.inputs,
        "tensor_keys": len(keys),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    (args.output / "weighted_ssl_backbone_meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()
