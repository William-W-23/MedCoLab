#!/usr/bin/env python3
"""Materialize one complete model state per client from a FedBN bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-clients", type=int, default=5)
    args = parser.parse_args()

    if not args.bundle.is_file():
        raise FileNotFoundError(args.bundle)
    payload = torch.load(args.bundle, map_location="cpu", weights_only=False)
    required = {"shared_state", "client_local_state", "round", "config", "metrics"}
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"FedBN bundle missing fields: {missing}")

    shared = payload["shared_state"]
    local_by_client = payload["client_local_state"]
    expected_ids = set(range(args.expected_clients))
    actual_ids = {int(key) for key in local_by_client}
    if actual_ids != expected_ids:
        raise RuntimeError(f"Expected clients {sorted(expected_ids)}, got {sorted(actual_ids)}")

    args.out_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for client_id in range(args.expected_clients):
        local = local_by_client.get(client_id, local_by_client.get(str(client_id)))
        overlap = sorted(set(shared) & set(local))
        if overlap:
            raise RuntimeError(f"Shared/local key overlap for client{client_id}: {overlap[:10]}")
        full_state = {name: tensor.detach().cpu().clone() for name, tensor in shared.items()}
        full_state.update({name: tensor.detach().cpu().clone() for name, tensor in local.items()})
        output = args.out_dir / f"client{client_id}_personalized_round0.pt"
        torch.save(full_state, output)
        records.append(
            {
                "client_id": client_id,
                "path": str(output),
                "sha256": sha256_file(output),
                "shared_tensors": len(shared),
                "local_tensors": len(local),
                "full_tensors": len(full_state),
            }
        )

    metadata = {
        "source_bundle": str(args.bundle),
        "source_bundle_sha256": sha256_file(args.bundle),
        "source_round": int(payload["round"]),
        "source_metrics": payload["metrics"],
        "num_clients": args.expected_clients,
        "clients": records,
    }
    (args.out_dir / "materialization_meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
