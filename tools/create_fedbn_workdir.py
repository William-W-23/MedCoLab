#!/usr/bin/env python3
"""Create an isolated Flower run directory without copying datasets or models."""
import argparse, os
from pathlib import Path
def main():
    p=argparse.ArgumentParser(); p.add_argument("--source-root",default=str(Path(__file__).resolve().parents[1])); p.add_argument("--workdir",required=True); p.add_argument("--serverapp",required=True); p.add_argument("--clientapp",required=True); a=p.parse_args()
    source=Path(a.source_root).resolve(); workdir=Path(a.workdir)
    if workdir.exists(): raise SystemExit(f"Refusing existing Flower workdir: {workdir}")
    workdir.mkdir(parents=True); lines=[]; text=(source/"pyproject.toml").read_text(encoding="utf-8")
    defaults={
        "classification_num_clients":"5", "classification_master_seed":"42",
        "classification_data_root":'"unused"', "classification_manifest_sha256":'"unused"',
        "classification_stratified_manifest":'"unused"', "classification_stratified_manifest_sha256":'"unused"',
        "classification_round0_path":'"unused"', "classification_output_dir":'"unused"',
        "classification_local_epochs":"1", "classification_lr":"0.00005",
        "classification_backbone_lr":"0.00001", "classification_head_lr":"0.00005",
        "classification_moe_lr":"0.00005", "classification_moe_enabled":"false",
        "classification_moe_num_experts":"4", "classification_moe_top_k":"2",
        "classification_moe_bottleneck":"256", "classification_moe_gamma_init":"0.001",
        "classification_moe_balance_loss_weight":"0.01",
        "classification_weight_decay":"0.0001", "classification_label_smoothing":"0.02",
        "classification_class_weight_power":"0.25", "classification_pcam_group_balanced":"true",
        "classification_group_audit_unavailable":"false",
        "classification_early_stop_macro_f1_threshold":"0.0",
        "classification_early_stop_min_rounds":"5",
        "classification_batch_size":"16", "classification_eval_batch_size":"32",
        "classification_image_size":"320", "classification_num_workers":"2",
        "classification_train_max_batches":"0", "classification_eval_max_batches":"0",
        "classification_train_max_samples":"0", "classification_eval_max_samples":"0",
        "use_fedbn":"false", "server_optimizer":'"FedAvg"', "server_eta":"0.002",
        "fedyogi_beta1":"0.8", "fedyogi_beta2":"0.95", "fedyogi_tau":"0.001",
        "fedbn_state_dir":'"unused"',
    }
    existing={line.split("=",1)[0].strip() for line in text.splitlines() if "=" in line and not line.lstrip().startswith("#")}
    for line in text.splitlines():
        stripped=line.strip()
        if stripped.startswith("serverapp ="): line=f'serverapp = "{a.serverapp}:app"'
        elif stripped.startswith("clientapp ="): line=f'clientapp = "{a.clientapp}:app"'
        lines.append(line)
        if stripped=="[tool.flwr.app.config]": lines.extend(f"{key} = {value}" for key,value in defaults.items() if key not in existing)
        if stripped=="[tool.flwr.federations.local-simulation]":
            if "options.backend.init-args.num-cpus" not in existing:
                lines.append("options.backend.init-args.num-cpus = 10")
            if "options.backend.init-args.num-gpus" not in existing:
                lines.append("options.backend.init-args.num-gpus = 1")
            if "options.backend.init-args.include-dashboard" not in existing:
                lines.append("options.backend.init-args.include-dashboard = false")
    (workdir/"pyproject.toml").write_text("\n".join(lines)+"\n",encoding="utf-8")
    for name in ("fl","models"): os.symlink(source/name,workdir/name,target_is_directory=True)
    (workdir/"SOURCE_ROOT").write_text(str(source)+"\n",encoding="utf-8")
if __name__=="__main__": main()
