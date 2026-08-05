"""Small helper for inspecting a MedCoLab checkpoint without starting training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def summarize_checkpoint(path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        print(f"type=dict keys={sorted(str(key) for key in payload)}")
        for key in ("round", "schema", "metrics", "config"):
            if key in payload:
                print(f"{key}={payload[key]}")
    else:
        print(f"type={type(payload).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    summarize_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
