#!/usr/bin/env python3
"""Conservative personalized classification fine-tuning with early stopping."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from fl.classification_task import (
    DATASET_CLASSES,
    DATASET_NAMES,
    MultiDatasetRTDETRClassifier,
    build_dataloader,
    confusions_from_metrics,
    evaluate,
    metrics_from_confusions,
    save_confusion_artifacts,
    seed_everything,
    sha256_file,
)
from fl.classification_client_support import build_stratified_loader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--client-model-root", type=Path)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--clients", default="0,1,2,3,4")
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--freeze-backbone-epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-delta", type=float, default=0.001)
    p.add_argument("--backbone-lr", type=float, default=1e-6)
    p.add_argument("--head-lr", type=float, default=1e-5)
    p.add_argument("--moe-lr", type=float, default=1e-5)
    p.add_argument("--moe-enabled", action="store_true")
    p.add_argument("--moe-num-experts", type=int, default=4)
    p.add_argument("--moe-top-k", type=int, default=2)
    p.add_argument("--moe-bottleneck", type=int, default=256)
    p.add_argument("--moe-gamma-init", type=float, default=1e-3)
    p.add_argument("--moe-balance-loss-weight", type=float, default=0.01)
    p.add_argument("--l2sp-mu", type=float, default=1e-3)
    p.add_argument("--class-weight-power", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=320)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--stratified-manifest", type=Path, required=True)
    p.add_argument("--stratified-manifest-sha256", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--pcam-group-balanced", action="store_true")
    p.add_argument("--dataset-equal-weight", type=float, default=0.3)
    p.add_argument("--max-dataset-f1-drop", type=float, default=0.02)
    p.add_argument("--soup-alphas", default="0,0.25,0.5,0.75,1")
    p.add_argument("--compact-sweep", action="store_true")
    p.add_argument("--group-audit-unavailable", action="store_true")
    return p.parse_args()


def clean(metrics):
    return {str(k): float(v) for k, v in metrics.items()
            if not str(k).startswith("cm__") and isinstance(v, (int, float))}


def class_weights(class_hist, dataset, power, device):
    counts = torch.tensor(
        [float(class_hist.get(f"{dataset}__{label}", 0)) for label in DATASET_CLASSES[dataset]],
        dtype=torch.float32, device=device,
    )
    weights = (counts.sum() / counts.clamp_min(1.0)).pow(power)
    return weights / weights.mean().clamp_min(1e-12)


def selection_score(metrics, dataset_equal_weight):
    dataset_macro = sum(float(metrics[f"{name}_macro_f1"]) for name in DATASET_NAMES) / len(DATASET_NAMES)
    weighted = float(metrics["weighted_f1"])
    weight = float(dataset_equal_weight)
    return (1.0 - weight) * weighted + weight * dataset_macro


def respects_dataset_guard(metrics, baseline, max_drop):
    return all(
        float(metrics[f"{name}_macro_f1"]) >= float(baseline[f"{name}_macro_f1"]) - float(max_drop)
        for name in DATASET_NAMES
    )


def interpolate_states(base_state, tuned_state, alpha):
    mixed = {}
    for name, base_value in base_state.items():
        tuned_value = tuned_state[name]
        if torch.is_floating_point(base_value):
            mixed[name] = base_value.mul(1.0 - alpha).add(tuned_value, alpha=alpha)
        else:
            mixed[name] = tuned_value.clone() if alpha > 0.0 else base_value.clone()
    return mixed


def train_epoch(model, loader, optimizer, reference, device, class_weight_power, l2sp_mu,
                backbone_frozen, moe_balance_loss_weight=0.01):
    model.train()
    if backbone_frozen:
        # Keep BatchNorm running statistics fixed together with frozen weights.
        model.backbone.eval()
    weights = {name: class_weights(loader.dataset.class_hist, name, class_weight_power, device)
               for name in DATASET_NAMES}
    loss_total = ce_total = prox_total = 0.0
    correct = seen = batches = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        dataset_ids = batch["dataset_id"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        features = model.encode(images)
        dataset_losses = []
        for dataset_id in dataset_ids.unique(sorted=True).tolist():
            mask = dataset_ids == dataset_id
            dataset = DATASET_NAMES[int(dataset_id)]
            logits = model.logits_for_dataset(features[mask], dataset_id)
            dataset_losses.append(F.cross_entropy(logits, labels[mask], weight=weights[dataset]))
            correct += int((logits.argmax(1) == labels[mask]).sum().item())
        ce_loss = torch.stack(dataset_losses).mean()
        if model.moe is not None:
            ce_loss = ce_loss + float(moe_balance_loss_weight) * model.moe.load_balance_loss()
        prox = ce_loss.new_zeros(())
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                prox = prox + (parameter - reference[name]).pow(2).sum()
        loss = ce_loss + 0.5 * l2sp_mu * prox
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_total += float(loss.item()); ce_total += float(ce_loss.item()); prox_total += float(prox.item())
        seen += int(labels.numel()); batches += 1
    return {
        "train_loss": loss_total / max(batches, 1),
        "train_ce_loss": ce_total / max(batches, 1),
        "train_l2sp_distance": prox_total / max(batches, 1),
        "train_accuracy": correct / max(seen, 1),
        "train_batches": batches, "trained_examples": seen,
    }


def save_curves(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(epochs, [r.get("train_loss", math.nan) for r in history], label="train")
    axes[0, 0].plot(epochs, [r["val_val_loss"] for r in history], label="val")
    axes[0, 0].set_title("Loss"); axes[0, 0].legend()
    axes[0, 1].plot(epochs, [r.get("train_accuracy", math.nan) for r in history], label="train")
    axes[0, 1].plot(epochs, [r["val_accuracy"] for r in history], label="val")
    axes[0, 1].set_title("Accuracy"); axes[0, 1].legend()
    axes[1, 0].plot(epochs, [r["val_weighted_f1"] for r in history])
    axes[1, 0].set_title("All-data val Weighted-F1")
    axes[1, 1].plot(epochs, [r["head_lr"] for r in history], label="head")
    axes[1, 1].plot(epochs, [r["backbone_lr"] for r in history], label="backbone")
    axes[1, 1].set_title("Learning rate"); axes[1, 1].legend()
    for ax in axes.flat: ax.set_xlabel("Epoch"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main():
    args = parse_args()
    if not args.model.is_file() or not args.data_root.is_dir():
        raise FileNotFoundError(f"model={args.model} data_root={args.data_root}")
    client_ids = [int(x) for x in args.clients.split(",") if x.strip()]
    base_state = torch.load(args.model, map_location="cpu", weights_only=False)
    def build_model():
        return MultiDatasetRTDETRClassifier(
            dropout=0.1, moe_enabled=args.moe_enabled,
            moe_num_experts=args.moe_num_experts, moe_top_k=args.moe_top_k,
            moe_bottleneck=args.moe_bottleneck, moe_gamma_init=args.moe_gamma_init,
        )
    probe = build_model(); probe.load_state_dict(base_state, strict=True); del probe
    print(f"Validated global checkpoint sha256={sha256_file(args.model)}", flush=True)
    if args.dry_run:
        for c in client_ids:
            for split in ("train", "val", "test"):
                _, stats = build_dataloader(args.data_root, c, split, args.image_size,
                                            args.eval_batch_size, args.seed, 0, 0)
                print(json.dumps(stats, sort_keys=True), flush=True)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "task": "medical5_classification_conservative_local_finetune_v10_pcamrobust",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "started_from_model": str(args.model), "started_from_sha256": sha256_file(args.model),
        "client_model_root": str(args.client_model_root) if args.client_model_root else None,
        "data_root": str(args.data_root), "manifest_sha256": args.manifest_sha256,
        "stratified_manifest": str(args.stratified_manifest), "stratified_manifest_sha256": args.stratified_manifest_sha256,
        "selection": "client_all_data_val_weighted_f1_on_group_safe_val_with_epoch0_global_fallback",
        "max_epochs": args.max_epochs, "freeze_backbone_epochs": args.freeze_backbone_epochs,
        "patience": args.patience, "min_delta": args.min_delta,
        "backbone_lr": args.backbone_lr, "head_lr": args.head_lr, "moe_lr": args.moe_lr,
        "moe_enabled": args.moe_enabled, "moe_num_experts": args.moe_num_experts,
        "moe_top_k": args.moe_top_k, "moe_bottleneck": args.moe_bottleneck,
        "moe_gamma_init": args.moe_gamma_init,
        "moe_balance_loss_weight": args.moe_balance_loss_weight,
        "l2sp_mu": args.l2sp_mu, "class_weight_power": args.class_weight_power,
        "loss_aggregation": "mean_of_dataset_specific_mean_losses_per_batch",
        "pcam_group_balanced": args.pcam_group_balanced,
        "dataset_equal_weight": args.dataset_equal_weight,
        "max_dataset_f1_drop": args.max_dataset_f1_drop,
        "soup_alphas": [float(x) for x in args.soup_alphas.split(",") if x.strip()],
        "pcam_augmentation": "full_field_resize_hflip_vflip_mild_colorjitter_no_random_resized_crop",
        "seed": args.seed, "client_seeds": {str(c): args.seed + c * 10000 for c in client_ids},
    }
    (args.out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, sort_keys=True))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results, raw_tests = [], []

    for client_id in client_ids:
        client_seed = args.seed + client_id * 10000
        seed_everything(client_seed)
        out = args.out_dir / f"client{client_id}"; out.mkdir(parents=True, exist_ok=True)
        client_base_state = base_state
        client_model_path = args.model
        if args.client_model_root:
            client_model = args.client_model_root / f"client{client_id}_personalized_round0.pt"
            if not client_model.is_file():
                raise FileNotFoundError(client_model)
            client_base_state = torch.load(client_model, map_location="cpu", weights_only=False)
            client_model_path = client_model
        model = build_model(); model.load_state_dict(client_base_state, strict=True); model.to(device)
        reference = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        backbone_params = list(model.backbone.parameters())
        head_params = list(model.heads.parameters())
        parameter_groups = [
            {"params": backbone_params, "lr": args.backbone_lr, "name": "backbone"},
            {"params": head_params, "lr": args.head_lr, "name": "heads"},
        ]
        if model.moe is not None:
            parameter_groups.append({"params": model.moe.parameters(), "lr": args.moe_lr, "name": "moe"})
        optimizer = torch.optim.AdamW(parameter_groups, weight_decay=1e-4)
        split_config={"classification_data_root":str(args.data_root),
            "classification_stratified_manifest":str(args.stratified_manifest),
            "classification_stratified_manifest_sha256":args.stratified_manifest_sha256,
            "classification_image_size":args.image_size,"classification_batch_size":args.batch_size,
            "classification_eval_batch_size":args.eval_batch_size,"classification_master_seed":args.seed,
            "classification_num_workers":args.num_workers,"classification_train_max_samples":0,"classification_eval_max_samples":0,
            "classification_pcam_group_balanced":args.pcam_group_balanced,
            "classification_group_audit_unavailable":args.group_audit_unavailable}
        train_loader, train_stats = build_stratified_loader(split_config, client_id, "train", 0)
        val_loader, val_stats = build_stratified_loader(split_config, client_id, "val", 0)
        test_loader, test_stats = build_stratified_loader(split_config, client_id, "test", 0)

        baseline_raw = evaluate(model, val_loader, device=device); baseline = clean(baseline_raw)
        best_f1 = float(baseline["weighted_f1"]); best_acc = float(baseline["accuracy"])
        best_score = selection_score(baseline, args.dataset_equal_weight)
        best_epoch = best_acc_epoch = 0
        best_path = out / "best_by_val_weighted_f1.pt"
        best_acc_path = out / "best_by_val_accuracy.pt"
        torch.save(model.state_dict(), best_path)
        if not args.compact_sweep:
            torch.save(model.state_dict(), best_acc_path)
        baseline_row = {
            "epoch": 0, "phase": "global_baseline", "selection_score": best_score,
            "dataset_guard_passed": True, **{f"val_{k}": v for k, v in baseline.items()}
        }
        (out / "best_by_val_weighted_f1_meta.json").write_text(json.dumps(baseline_row, indent=2))
        (out / "best_by_val_accuracy_meta.json").write_text(json.dumps(baseline_row, indent=2))
        history = []
        no_improvement = 0
        print(f"=== client{client_id} baseline val_f1={best_f1:.6f} val_acc={best_acc:.6f} ===", flush=True)

        for epoch in range(1, args.max_epochs + 1):
            backbone_trainable = epoch > args.freeze_backbone_epochs
            for parameter in backbone_params: parameter.requires_grad = backbone_trainable
            progress = (epoch - 1) / max(args.max_epochs - 1, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            scheduled_backbone_lr = 1e-7 + (args.backbone_lr - 1e-7) * cosine
            scheduled_head_lr = 1e-7 + (args.head_lr - 1e-7) * cosine
            optimizer.param_groups[0]["lr"] = scheduled_backbone_lr if backbone_trainable else 0.0
            optimizer.param_groups[1]["lr"] = scheduled_head_lr
            if model.moe is not None:
                optimizer.param_groups[2]["lr"] = 1e-7 + (args.moe_lr - 1e-7) * cosine
            train_metrics = train_epoch(model, train_loader, optimizer, reference, device,
                                        args.class_weight_power, args.l2sp_mu,
                                        backbone_frozen=not backbone_trainable,
                                        moe_balance_loss_weight=args.moe_balance_loss_weight)
            val_raw = evaluate(model, val_loader, device=device); val = clean(val_raw)
            row = {
                "epoch": epoch, "backbone_frozen": not backbone_trainable,
                "backbone_lr": optimizer.param_groups[0]["lr"], "head_lr": optimizer.param_groups[1]["lr"],
                **train_metrics, **{f"val_{k}": v for k, v in val.items()},
            }
            pcam_f1, pcam_acc = float(val["weighted_f1"]), float(val["accuracy"])
            score = selection_score(val, args.dataset_equal_weight)
            guard_passed = respects_dataset_guard(val, baseline, args.max_dataset_f1_drop)
            row["selection_score"] = score
            row["dataset_guard_passed"] = guard_passed
            history.append(row); (out / "metrics_history.json").write_text(json.dumps(history, indent=2))
            improved = guard_passed and score > best_score + args.min_delta
            if improved:
                best_f1, best_score, best_epoch, no_improvement = pcam_f1, score, epoch, 0
                torch.save(model.state_dict(), best_path)
                (out / "best_by_val_weighted_f1_meta.json").write_text(json.dumps(row, indent=2))
            else:
                no_improvement += 1
            if pcam_acc > best_acc + args.min_delta:
                best_acc, best_acc_epoch = pcam_acc, epoch
                if not args.compact_sweep:
                    torch.save(model.state_dict(), best_acc_path)
                    (out / "best_by_val_accuracy_meta.json").write_text(json.dumps(row, indent=2))
            print(f"[client {client_id}] epoch {epoch}/{args.max_epochs} train={train_metrics['train_loss']:.6f} "
                  f"val_f1={pcam_f1:.6f} score={score:.6f} guard={guard_passed} "
                  f"best_score={best_score:.6f}@{best_epoch} patience={no_improvement}/{args.patience}", flush=True)
            if no_improvement >= args.patience:
                print(f"[client {client_id}] early stop at epoch {epoch}", flush=True); break

        if not args.compact_sweep:
            torch.save(model.state_dict(), out / "final_model.pt")
            save_curves(history, out / "convergence_curves.png")
        tuned_state = torch.load(best_path, map_location="cpu", weights_only=False)
        soup_candidates = []
        for alpha in [float(x) for x in args.soup_alphas.split(",") if x.strip()]:
            mixed_state = interpolate_states(client_base_state, tuned_state, alpha)
            model.load_state_dict(mixed_state, strict=True)
            soup_val = clean(evaluate(model, val_loader, device=device))
            soup_score = selection_score(soup_val, args.dataset_equal_weight)
            soup_guard = respects_dataset_guard(soup_val, baseline, args.max_dataset_f1_drop)
            soup_candidates.append({
                "alpha": alpha, "selection_score": soup_score,
                "weighted_f1": float(soup_val["weighted_f1"]),
                "accuracy": float(soup_val["accuracy"]), "dataset_guard_passed": soup_guard,
            })
        valid_soups = [row for row in soup_candidates if row["dataset_guard_passed"]]
        selected_soup = max(valid_soups, key=lambda row: row["selection_score"])
        selected_state = interpolate_states(client_base_state, tuned_state, float(selected_soup["alpha"]))
        model.load_state_dict(selected_state, strict=True)
        torch.save(selected_state, best_path)
        (out / "soup_selection.json").write_text(json.dumps({
            "selection_uses_validation_only": True,
            "candidates": soup_candidates, "selected": selected_soup,
        }, indent=2))
        best_f1 = float(selected_soup["weighted_f1"])
        (out / "best_by_val_weighted_f1_meta.json").write_text(json.dumps({
            "epoch": best_epoch, "phase": "validation_selected_global_local_soup",
            "selected_soup_alpha": float(selected_soup["alpha"]),
            "selection_score": float(selected_soup["selection_score"]),
            "dataset_guard_passed": bool(selected_soup["dataset_guard_passed"]),
            "val_weighted_f1": best_f1, "val_accuracy": float(selected_soup["accuracy"]),
        }, indent=2))
        test_raw = None if args.skip_test else evaluate(model, test_loader, device=device)
        if test_raw is not None:
            raw_tests.append(test_raw); test_metrics = clean(test_raw)
            save_confusion_artifacts(confusions_from_metrics([test_raw]), out, "test_best")
        else: test_metrics = {}
        result = {
            "client_id": client_id, "client_seed": client_seed,
            "started_from_model": str(client_model_path),
            "started_from_sha256": sha256_file(client_model_path),
            "baseline_val_weighted_f1": float(baseline["weighted_f1"]),
            "best_epoch": best_epoch, "best_val_weighted_f1": best_f1,
            "best_accuracy_epoch": best_acc_epoch, "best_val_accuracy": best_acc,
            "accepted_local_update": float(selected_soup["alpha"]) > 0.0,
            "selected_soup_alpha": float(selected_soup["alpha"]),
            "selection_score": float(selected_soup["selection_score"]),
            "epochs_run": len(history), "stopped_early": len(history) < args.max_epochs,
            "best_model": str(best_path), "best_model_sha256": sha256_file(best_path),
            "train": train_stats, "val": val_stats, "test": test_stats, "test_metrics": test_metrics,
        }
        (out / "test_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True))
        results.append(result)
        if test_metrics:
            print(f"[client {client_id}] TEST weighted_f1={test_metrics['weighted_f1']:.6f} accuracy={test_metrics['accuracy']:.6f}", flush=True)
        del model, optimizer, reference; torch.cuda.empty_cache()

    combined = {}
    if raw_tests:
        confusions = confusions_from_metrics(raw_tests); combined = metrics_from_confusions(confusions)
        total = sum(float(x["eval_examples"]) for x in raw_tests)
        combined["top5_accuracy"] = sum(float(x["top5_accuracy"]) * float(x["eval_examples"]) for x in raw_tests) / max(total, 1)
        save_confusion_artifacts(confusions, args.out_dir, "combined_test")
    summary = {**run_meta, "completed_at": datetime.now(timezone.utc).isoformat(),
               "clients": results, "combined_test_metrics": combined}
    (args.out_dir / "local_finetune_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    with (args.out_dir / "client_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["client", "best_epoch", "epochs_run", "accuracy", "weighted_f1", "macro_f1"])
        for row in results:
            m = row["test_metrics"]; writer.writerow([row["client_id"], row["best_epoch"], row["epochs_run"], m.get("accuracy",""), m.get("weighted_f1",""), m.get("macro_f1","")])


if __name__ == "__main__":
    main()
