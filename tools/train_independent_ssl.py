#!/usr/bin/env python3
"""Train one independent MoCo SSL client without Flower/Ray aggregation."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

from fl.ssl_task import (
    SSL_DEFAULTS,
    _save_ssl_artifact_pair,
    build_mask_config,
    build_ssl_dataloader,
    create_ssl_model,
    load_pretrained_backbone,
    load_ssl_full_checkpoint,
    save_ssl_server_artifacts,
    seed_everything,
    ssl_train_one_round,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--source-client", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-variant", default="RTDETR_L")
    parser.add_argument("--pretrained", default="weights/rtdetr-l.pt")
    parser.add_argument("--mask-adaptive", action="store_true")
    parser.add_argument("--mean-samples", type=float, default=0.0)
    return parser.parse_args()


def save_history(path: Path, history: list[dict]) -> None:
    path.write_text(json.dumps(history, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(SSL_DEFAULTS)
    config.update(
        {
            "ssl_dataset": args.dataset_name,
            "ssl_dataset_root": args.data_root,
            "ssl_dataset_roots": "",
            "ssl_balance_datasets_per_batch": True,
            "ssl_num_clients": 1,
            "ssl_partition_method": "fixed_client_dirs",
            "ssl_model_variant": args.model_variant,
            "ssl_local_epochs": args.local_epochs,
            "ssl_batch_size": args.batch_size,
            "ssl_train_max_batches": args.max_batches,
            "ssl_lr": args.lr,
            "ssl_image_size": args.image_size,
            "ssl_pretrained_detector_ckpt": args.pretrained,
            "num-server-rounds": args.rounds,
            "master_seed": args.seed,
            "source_client": args.source_client,
            "data_manifest_sha256": args.manifest_sha256,
            "code_version": os.environ.get("CODE_VERSION", ""),
            "started_at": os.environ.get("EXPERIMENT_STARTED_AT", datetime.now(timezone.utc).isoformat()),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "training_scope": "independent_client_ssl_no_aggregation",
            "ssl_mask_adaptive": args.mask_adaptive,
        }
    )
    config["mode_signature"] = (
        f"task:{config['ssl_mode']}|model:{args.model_variant}|scope:{config['ssl_pretrain_scope']}|"
        f"strategy:Independent|dataset:{args.dataset_name}|client:{args.source_client}"
    )

    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_ssl_model(config, device)
    load_report = load_pretrained_backbone(model, args.pretrained)

    history_path = args.output / "ssl_metrics_history.json"
    resume_path = args.output / "ssl_resume_last_full.pt"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    start_round = int(history[-1]["round"]) + 1 if history else 1
    if start_round > 1:
        if not resume_path.exists():
            raise FileNotFoundError(f"Missing resume checkpoint for round {start_round}: {resume_path}")
        load_report = {"pretrained": load_report, "resume": load_ssl_full_checkpoint(model, str(resume_path))}

    best_loss = math.inf
    best_round = None
    valid_history = [row for row in history if row.get("ssl_loss") is not None]
    if valid_history:
        best_row = min(valid_history, key=lambda row: float(row["ssl_loss"]))
        best_loss = float(best_row["ssl_loss"])
        best_round = int(best_row["round"])

    for round_num in range(start_round, args.rounds + 1):
        config["_round_seed_offset"] = round_num
        seed_everything(args.seed + round_num * 10000)
        trainloader, stats = build_ssl_dataloader(0, config)
        mask_config = build_mask_config(
            config,
            round_num=round_num,
            total_rounds=args.rounds,
            num_samples=int(stats["num_samples"]),
            mean_samples=(args.mean_samples if args.mean_samples > 0 else float(stats["num_samples"])),
        )
        metrics = ssl_train_one_round(
            model,
            trainloader,
            local_epochs=args.local_epochs,
            lr=args.lr,
            momentum=float(config["ssl_momentum"]),
            device=device,
            mask_config=mask_config,
            max_batches=args.max_batches,
        )
        loss = float(metrics["ssl_loss"])
        row = {"round": round_num, **{key: float(value) for key, value in metrics.items()}}
        history.append(row)
        save_history(history_path, history)

        _save_ssl_artifact_pair(
            model,
            config,
            args.output,
            backbone_name="ssl_resume_last_backbone.pt",
            full_name="ssl_resume_last_full.pt",
            meta_name="ssl_resume_last_meta.json",
            server_round=round_num,
        )
        if math.isfinite(loss) and loss < best_loss:
            best_loss = loss
            best_round = round_num
            _save_ssl_artifact_pair(
                model,
                config,
                args.output,
                backbone_name="ssl_best_by_loss_backbone.pt",
                full_name="ssl_best_by_loss_full.pt",
                meta_name="ssl_best_by_loss_meta.json",
                server_round=round_num,
            )
            best_meta_path = args.output / "ssl_best_by_loss_meta.json"
            best_meta = json.loads(best_meta_path.read_text())
            best_meta["best_ssl_loss"] = best_loss
            best_meta_path.write_text(json.dumps(best_meta, indent=2) + "\n")
        print(
            f"[INDEPENDENT SSL {args.source_client}] round={round_num}/{args.rounds} "
            f"ssl_loss={loss:.8f} best_round={best_round} best_loss={best_loss:.8f}",
            flush=True,
        )

    save_ssl_server_artifacts(model, config, args.output)
    (args.output / "server_result.json").write_text(
        json.dumps(
            {
                "mode_signature": config["mode_signature"],
                "rounds": args.rounds,
                "source_client": args.source_client,
                "master_seed": args.seed,
                "manifest_sha256": args.manifest_sha256,
                "load_report": load_report,
                "best_round": best_round,
                "best_ssl_loss": best_loss,
            },
            indent=2,
        )
        + "\n"
    )
    for name in ("ssl_resume_last_backbone.pt", "ssl_resume_last_full.pt", "ssl_resume_last_meta.json"):
        (args.output / name).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
