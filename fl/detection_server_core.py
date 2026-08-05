"""FL: A Flower / PyTorch app."""

import torch
import logging
import os
import random
import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedYogi
from flwr.common.logger import log

# ============================================================================
# 模型选择配置 - 两种模式
# ============================================================================
# 模式 A: RTDETR_L（标准版，无 RetNet）
# 模式 B: RTDETR_L_WithAttention（带 RetNet Attention）
# 模式 C: RTDETR_L_WithASEM（解剖感知稀疏 MoE）

# 通过环境变量切换模型，默认保持 ASEM，避免手动改 server/client 后忘记同步。
from models import RTDETR_L, RTDETR_L_WithAttention, RTDETR_L_WithASEM

_MODEL_VARIANTS = {
    "RTDETR_L": RTDETR_L,
    "RTDETR_L_WithAttention": RTDETR_L_WithAttention,
    "RTDETR_L_WithASEM": RTDETR_L_WithASEM,
}
FL_MODEL_VARIANT = os.environ.get("FL_MODEL_VARIANT", "RTDETR_L_WithASEM")
if FL_MODEL_VARIANT not in _MODEL_VARIANTS:
    raise ValueError(f"Unsupported FL_MODEL_VARIANT={FL_MODEL_VARIANT!r}; choose one of {sorted(_MODEL_VARIANTS)}")
Net = _MODEL_VARIANTS[FL_MODEL_VARIANT]

# ============================================================================
# 策略选择配置
# ============================================================================
# 模式 1: FedAvg（标准聚合，发送完整 state_dict）
# 模式 2: FedYogi + FedBN（自适应聚合，只发送非 BN 参数）

USE_FEDBN = False  # True: FedBN + FedYogi; False: FedAvg

# ============================================================================
# 预训练权重路径
# ============================================================================
pretrained_model_path = os.environ.get("FL_PRETRAINED_MODEL_PATH", "weights/rtdetr-l.pt")
ssl_backbone_path = os.environ.get("FL_SSL_BACKBONE_CKPT", "")

# ============================================================================
# 输出目录配置（按模式隔离）
# ============================================================================
from pathlib import Path
from fl.finetune_from_ssl_task import load_ssl_backbone_weights
from fl.detection_task import (
    load_rtdetr_weights,
    get_federated_model_record,
    load_federated_model_record,
    load_model_arrays,
    describe_fedbn_layout,
    get_mode_signature,
    set_current_mode,
    extract_model_signature,
    DATASET_CONFIGS,
    CURRENT_DATASET,
    FEDBN_STATE_DIR,
)

# 创建 ServerApp
app = ServerApp()




def parse_checkpoint_rounds(value: str) -> set[int]:
    """Parse comma-separated round numbers for server-side raw-array checkpoints."""
    rounds: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            round_id = int(item)
        except ValueError:
            log(logging.WARNING, f"Ignoring invalid checkpoint round: {item}")
            continue
        if round_id > 0:
            rounds.add(round_id)
    return rounds


def _metric_record_to_plain(record):
    return {str(k): float(v) if isinstance(v, (int, float)) else v for k, v in dict(record).items()}


def _history_to_plain(history):
    return {str(k): _metric_record_to_plain(v) for k, v in dict(history).items()}


def _round_metric(hist, round_id, key):
    value = hist.get(str(round_id), {}).get(key)
    return None if value is None else float(value)


def _best_round_by_metric(eval_hist, key):
    candidates = []
    for round_id, metrics in eval_hist.items():
        value = metrics.get(key)
        if value is not None:
            candidates.append((int(round_id), float(value)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])[0]


def save_metrics_artifacts(result, output_dir: Path) -> None:
    """Save compact metrics artifacts for convergence and reporting."""
    import json
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    train_hist = _history_to_plain(result.train_metrics_clientapp)
    eval_hist = _history_to_plain(result.evaluate_metrics_clientapp)
    payload = {
        "train_metrics_clientapp": train_hist,
        "evaluate_metrics_clientapp": eval_hist,
        "evaluate_metrics_serverapp": _history_to_plain(result.evaluate_metrics_serverapp),
    }
    metrics_json = metrics_dir / "metrics_history.json"
    with open(metrics_json, "w") as f:
        json.dump(payload, f, indent=2)
    log(logging.INFO, f"Metrics history saved to: {metrics_json}")

    eval_rounds = sorted(int(r) for r in eval_hist.keys())
    train_rounds = sorted(int(r) for r in train_hist.keys())
    final_round = eval_rounds[-1] if eval_rounds else (train_rounds[-1] if train_rounds else 0)
    best_map_round = _best_round_by_metric(eval_hist, "eval_map50")
    best_f1_round = _best_round_by_metric(eval_hist, "eval_f1")

    def block(title, round_id):
        if round_id is None:
            return [f"{title}: N/A"]
        return [
            f"{title}:",
            f"Round: {round_id}",
            f"mAP50: {_round_metric(eval_hist, round_id, 'eval_map50'):.4f}" if _round_metric(eval_hist, round_id, 'eval_map50') is not None else "mAP50: N/A",
            f"Precision: {_round_metric(eval_hist, round_id, 'eval_precision'):.4f}" if _round_metric(eval_hist, round_id, 'eval_precision') is not None else "Precision: N/A",
            f"Recall: {_round_metric(eval_hist, round_id, 'eval_recall'):.4f}" if _round_metric(eval_hist, round_id, 'eval_recall') is not None else "Recall: N/A",
            f"F1: {_round_metric(eval_hist, round_id, 'eval_f1'):.4f}" if _round_metric(eval_hist, round_id, 'eval_f1') is not None else "F1: N/A",
            f"eval_loss: {_round_metric(eval_hist, round_id, 'eval_loss'):.4f}" if _round_metric(eval_hist, round_id, 'eval_loss') is not None else "eval_loss: N/A",
            f"train_loss: {_round_metric(train_hist, round_id, 'train_loss'):.4f}" if _round_metric(train_hist, round_id, 'train_loss') is not None else "train_loss: N/A",
        ]

    summary_lines = []
    summary_lines.extend(block("Best mAP50", best_map_round))
    summary_lines.append("")
    summary_lines.extend(block("Best F1", best_f1_round))
    summary_lines.append("")
    summary_lines.extend(block("Final Round", final_round if final_round else None))
    summary_path = metrics_dir / "summary_metrics.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    log(logging.INFO, f"Summary metrics saved to: {summary_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log(logging.WARNING, f"Skipping metrics curves; matplotlib unavailable: {exc}")
        return

    rounds = sorted(int(r) for r in eval_hist.keys())
    if not rounds:
        return

    def values(hist, key):
        return [hist.get(str(r), {}).get(key, None) for r in rounds]

    train_rounds = sorted(int(r) for r in train_hist.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    if train_rounds:
        axes[0, 0].plot(train_rounds, [train_hist[str(r)].get("train_loss") for r in train_rounds], label="train_loss", color="#1f77b4")
    axes[0, 0].plot(rounds, values(eval_hist, "eval_loss"), label="eval_loss", color="#d62728")
    axes[0, 0].set_title("Loss Curve")
    axes[0, 0].set_xlabel("Round")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(rounds, values(eval_hist, "eval_map50"), label="mAP50", color="#2ca02c")
    axes[0, 1].set_title("mAP50 Curve")
    axes[0, 1].set_xlabel("Round")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(rounds, values(eval_hist, "eval_precision"), label="precision", color="#9467bd")
    axes[1, 0].plot(rounds, values(eval_hist, "eval_recall"), label="recall", color="#ff7f0e")
    axes[1, 0].plot(rounds, values(eval_hist, "eval_f1"), label="f1", color="#17becf")
    axes[1, 0].set_title("Precision / Recall / F1")
    axes[1, 0].set_xlabel("Round")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    map_vals = values(eval_hist, "eval_map50")
    f1_vals = values(eval_hist, "eval_f1")
    if map_vals and any(v is not None for v in map_vals):
        best_map_idx = max(range(len(map_vals)), key=lambda i: map_vals[i] if map_vals[i] is not None else -1)
        axes[1, 1].scatter([rounds[best_map_idx]], [map_vals[best_map_idx]], color="#2ca02c", label=f"best mAP50 r{rounds[best_map_idx]}")
    axes[1, 1].plot(rounds, map_vals, label="mAP50", color="#2ca02c", alpha=0.7)
    axes[1, 1].plot(rounds, f1_vals, label="F1", color="#17becf", alpha=0.7)
    axes[1, 1].set_title("Best-Round Check")
    axes[1, 1].set_xlabel("Round")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    curves_path = metrics_dir / "convergence_curves.png"
    fig.savefig(curves_path, dpi=180)
    plt.close(fig)
    log(logging.INFO, f"Convergence curves saved to: {curves_path}")


class CheckpointingFedAvg(FedAvg):
    """FedAvg with server-side raw ArrayRecord checkpoints and best-mAP saving."""

    def __init__(self, *args, checkpoint_dir: Path | None = None, checkpoint_rounds: set[int] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_rounds = checkpoint_rounds or set()
        self.latest_arrays = None
        self.best_metric_name = "eval_map50"
        self.best_metric_value = float("-inf")
        self.best_metric_round = 0
        self.best_arrays_path = None

    def _save_raw_arrays(self, arrays, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(arrays.to_torch_state_dict(), path)

    def aggregate_train(self, server_round, replies):
        arrays, metrics = super().aggregate_train(server_round, replies)
        if arrays is not None:
            self.latest_arrays = arrays
            if self.checkpoint_dir is not None and server_round in self.checkpoint_rounds:
                checkpoint_path = self.checkpoint_dir / f"global_arrays_round_{server_round}（临时）.pt"
                self._save_raw_arrays(arrays, checkpoint_path)
                log(logging.INFO, f"Saved server raw-array checkpoint: {checkpoint_path}")
        return arrays, metrics

    def aggregate_evaluate(self, server_round, replies):
        metrics = super().aggregate_evaluate(server_round, replies)
        if metrics is not None and self.latest_arrays is not None and self.checkpoint_dir is not None:
            metric_value = dict(metrics).get(self.best_metric_name)
            if metric_value is not None:
                metric_value = float(metric_value)
                if metric_value > self.best_metric_value:
                    self.best_metric_value = metric_value
                    self.best_metric_round = int(server_round)
                    self.best_arrays_path = self.checkpoint_dir.parent / "best_model_by_map50_arrays.pt"
                    self._save_raw_arrays(self.latest_arrays, self.best_arrays_path)
                    import json
                    best_meta_path = self.checkpoint_dir.parent / "best_model_by_map50_meta.json"
                    with open(best_meta_path, "w") as f:
                        json.dump({
                            "best_metric": self.best_metric_name,
                            "best_value": self.best_metric_value,
                            "best_round": self.best_metric_round,
                            "metrics": _metric_record_to_plain(metrics),
                            "arrays_path": str(self.best_arrays_path),
                        }, f, indent=2)
                    log(logging.INFO, f"Saved best-mAP50 raw arrays: {self.best_arrays_path} (round={server_round}, mAP50={metric_value:.6f})")
        return metrics

def get_output_dir(model_variant: str, strategy: str, dataset: str) -> Path:
    """
    根据模式生成独立的输出目录，避免状态串用。
    """
    base_dir = Path("outputs")
    experiment_tag = os.environ.get("FL_OUTPUT_TAG", "").strip()
    output_name = f"{model_variant}_{strategy}_{dataset}"
    if experiment_tag:
        output_name = f"{output_name}_{experiment_tag}"
    output_dir = base_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    master_seed = int(os.environ.get("MASTER_SEED", "42"))
    random.seed(master_seed)
    np.random.seed(master_seed)
    torch.manual_seed(master_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(master_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]

    # Get the correct number of classes for the current dataset
    nc = DATASET_CONFIGS[CURRENT_DATASET]["nc"]

    # ============================================================================
    # 设置模式签名（用于状态隔离和校验）
    # ============================================================================
    model_variant = Net.__name__  # 'RTDETR_L' 或 'RTDETR_L_WithAttention'
    strategy_name = "FedYogi" if USE_FEDBN else "FedAvg"
    dataset_name = CURRENT_DATASET

    mode_signature = get_mode_signature(
        model_variant=model_variant,
        strategy=strategy_name,
        dataset=dataset_name,
        nc=nc
    )
    set_current_mode(mode_signature)

    # 设置输出目录（按模式隔离）
    output_dir = get_output_dir(model_variant, strategy_name, dataset_name)

    log(logging.INFO,
        f"Server starting with mode signature: {mode_signature}")
    log(logging.INFO,
        f"Output directory: {output_dir}")
    log(logging.INFO,
        f"Model variant: {model_variant}, Strategy: {strategy_name}, "
        f"Dataset: {dataset_name}, nc={nc}, FedBN: {USE_FEDBN}")

    # Load global model with correct nc
    global_model = Net(nc=nc)
    global_model = load_rtdetr_weights(global_model, pretrained_model_path)
    ssl_loaded = 0
    if ssl_backbone_path:
        ssl_loaded = load_ssl_backbone_weights(global_model, ssl_backbone_path)
        log(logging.INFO, f"Loaded SSL backbone weights: {ssl_loaded} keys from {ssl_backbone_path}")

    # 打印模型签名
    model_sig = extract_model_signature(global_model)
    log(logging.INFO, f"Global model signature: {model_sig}")

    freeze_layers = range(10)  # 对应 Backbone (HGStem 到 Stage 4)

    # ============================================================================
    # 初始化 arrays（根据策略选择）
    # ============================================================================
    if USE_FEDBN:
        # FedBN + FedYogi 模式：只发送非 BN 参数
        arrays = get_federated_model_record(global_model)
        log(logging.INFO,
            "FedBN mode: arrays contains only non-BN federated params")
    else:
        # FedAvg 模式：发送完整 state_dict（包括 buffer）
        arrays = ArrayRecord(global_model.state_dict())
        log(logging.INFO,
            f"FedAvg mode: arrays contains full state_dict ({len(arrays)} keys)")

    print(
        "FedBN server init: "
        f"{describe_fedbn_layout(global_model)}"
    )

    # ============================================================================
    # 初始化策略
    # ============================================================================
    checkpoint_rounds = parse_checkpoint_rounds(os.environ.get("FL_CHECKPOINT_ROUNDS", ""))
    if checkpoint_rounds:
        log(logging.INFO, f"Server raw-array checkpoint rounds: {sorted(checkpoint_rounds)}")

    if USE_FEDBN:
        strategy = FedYogi(
            fraction_train=fraction_train,
            eta=2e-3,
            tau=1e-3,
            beta_1=0.8,
            beta_2=0.95,
        )
    else:
        strategy = CheckpointingFedAvg(
            fraction_train=fraction_train,
            checkpoint_dir=output_dir / "checkpoints",
            checkpoint_rounds=checkpoint_rounds,
        )

    # ============================================================================
    # 构造 train_config（包含模式签名，传递给 client）
    # ============================================================================
    train_config = ConfigRecord({
        "lr": lr,
        "mode_signature": mode_signature,
        "model_variant": model_variant,
        "strategy": strategy_name,
        "dataset": dataset_name,
        "nc": nc,
        "use_fedbn": USE_FEDBN,
    })

    log(logging.INFO, f"Train config: {dict(train_config)}")

    # Start strategy, run for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=train_config,
        num_rounds=num_rounds,
    )

    save_metrics_artifacts(result, output_dir)

    # ============================================================================
    # Save final model to disk（按模式隔离）
    # ============================================================================
    final_model_path = output_dir / "final_model.pt"

    log(logging.INFO, f"\nSaving final model to: {final_model_path}")

    final_arrays_path = output_dir / "final_arrays.pt"
    torch.save(result.arrays.to_torch_state_dict(), final_arrays_path)
    log(logging.INFO, f"Final raw arrays saved to: {final_arrays_path}")

    final_model = Net(nc=nc)
    final_model = load_rtdetr_weights(final_model, pretrained_model_path)
    if ssl_backbone_path:
        try:
            load_ssl_backbone_weights(final_model, ssl_backbone_path)
        except FileNotFoundError as exc:
            log(logging.WARNING, f"SSL backbone missing during final reconstruction, continuing with aggregated arrays: {exc}")

    # 使用增强的加载函数（允许宽松加载，因为这是最终保存）
    load_model_arrays(final_model, result.arrays, strict=False)

    torch.save(final_model.state_dict(), final_model_path)

    best_arrays_path = getattr(strategy, "best_arrays_path", None)
    if best_arrays_path is not None and Path(best_arrays_path).exists():
        best_model_path = output_dir / "best_model_by_map50.pt"
        best_model = Net(nc=nc)
        best_model = load_rtdetr_weights(best_model, pretrained_model_path)
        if ssl_backbone_path:
            try:
                load_ssl_backbone_weights(best_model, ssl_backbone_path)
            except FileNotFoundError as exc:
                log(logging.WARNING, f"SSL backbone missing during best model reconstruction, continuing with best arrays: {exc}")
        best_state = torch.load(best_arrays_path, map_location="cpu", weights_only=False)
        best_model.load_state_dict(best_state, strict=False)
        torch.save(best_model.state_dict(), best_model_path)
        log(logging.INFO, f"Best mAP50 model saved to: {best_model_path}")
        try:
            Path(best_arrays_path).unlink()
            log(logging.INFO, f"Removed temporary best arrays after best model save: {best_arrays_path}")
        except OSError as exc:
            log(logging.WARNING, f"Could not remove temporary best arrays: {exc}")

    # 保存模式元信息
    meta_path = output_dir / "mode_meta.json"
    import json
    with open(meta_path, 'w') as f:
        json.dump({
            "mode_signature": mode_signature,
            "model_variant": model_variant,
            "strategy": strategy_name,
            "dataset": dataset_name,
            "nc": nc,
            "num_rounds": num_rounds,
            "use_fedbn": USE_FEDBN,
            "pretrained_model_path": pretrained_model_path,
            "ssl_backbone_path": ssl_backbone_path,
            "ssl_backbone_keys_loaded": ssl_loaded,
            "master_seed": master_seed,
            "data_manifest_sha256": os.environ.get("DATA_MANIFEST_SHA256", ""),
            "ssl_round0_sha256": os.environ.get("SSL_ROUND0_SHA256", ""),
            "code_version": os.environ.get("CODE_VERSION", ""),
            "experiment_started_at": os.environ.get("EXPERIMENT_STARTED_AT", ""),
            "python_version": os.sys.version,
            "torch_version": torch.__version__,
        }, f, indent=2)

    log(logging.INFO, f"Mode metadata saved to: {meta_path}")
    log(logging.INFO, "Training completed successfully!")
